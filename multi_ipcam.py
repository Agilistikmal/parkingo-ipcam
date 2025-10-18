import easyocr
import cv2
import numpy as np
import re
import threading
from queue import Queue, Empty
import time
import json
import logging
from typing import Dict, Any, List, Tuple
from validate import validate_booking_plate, play_valid_sound, play_invalid_sound

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MultiCamLPR')

LANGUAGES = ['en'] 
ALLOW_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
PROCESS_INTERVAL_SEC = 1

global_camera_states: Dict[str, Dict[str, Any]] = {}
global_state_lock = threading.Lock()

try:
    reader = easyocr.Reader(LANGUAGES, gpu=False) 
    logger.info("EasyOCR Reader siap digunakan (CPU Mode).")
except Exception as e:
    logger.error(f"Gagal inisialisasi EasyOCR: {e}")
    exit()

def clean_license_plate(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^A-Z0-9 ]', '', text.upper()) 
    
    match = re.match(r'^([A-Z]{1,2})\s*([0-9]{1,4})\s*([A-Z]{0,3})$', text)
    
    if match:
        area_code, numbers, suffix = match.groups()
        area_code = area_code.replace('8', 'B').replace('0', 'O')
        numbers = numbers.replace('O', '0').replace('I', '1')
        suffix = suffix.replace('0', 'O')
        
        parts = [area_code, numbers]
        if suffix: parts.append(suffix)
            
        return ' '.join(parts)
    return ""

def scan_plate_simple(frame: np.ndarray) -> List[Tuple[str, List]]:
    if frame is None or frame.size == 0:
        return []

    found_plates_with_boxes = []
    
    try:
        results = reader.readtext(frame, allowlist=ALLOW_LIST)
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0])) 

        full_text_segments = []
        combined_bbox_points = []
        
        for (bbox_corners, text, confidence) in results:
            if confidence > 0.5 and text.strip(): 
                full_text_segments.append(text)
                combined_bbox_points.extend(bbox_corners)
                
        full_raw_text = ' '.join(full_text_segments)

        if not full_raw_text:
            return []

        cleaned_plate = clean_license_plate(full_raw_text)

        if cleaned_plate and combined_bbox_points:
            x_coords = [p[0] for p in combined_bbox_points]
            y_coords = [p[1] for p in combined_bbox_points]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            
            found_plates_with_boxes.append((cleaned_plate, [x, y, w, h]))
            
        return found_plates_with_boxes
            
    except Exception as e:
        logger.debug(f"Error saat memproses frame: {e}")
        return []

def ocr_worker(slot_id: str):
    while True:
        try:
            with global_state_lock:
                frame_queue: Queue = global_camera_states[slot_id]['frame_queue']
                last_validated_plate = global_camera_states[slot_id].get('last_validated_plate', None)

            frame = frame_queue.get_nowait() 
            
            if frame is None:
                break

            start_time = time.time()
            plates_with_boxes = scan_plate_simple(frame)
            end_time = time.time()
            
            with global_state_lock:
                if plates_with_boxes:
                    new_plate, bbox_coords = plates_with_boxes[0]
                    if new_plate != last_validated_plate:
                        validated_data = validate_booking_plate(slot_id, new_plate)
                        if validated_data:
                            is_valid = validated_data['is_valid']
                            similarity = validated_data['similarity']
                            
                            global_camera_states[slot_id]['last_validated_plate'] = new_plate
                            updated_plate_info = (new_plate, is_valid, similarity, bbox_coords)
                            plates_with_boxes[0] = updated_plate_info 
                        else:
                            logger.info(f"Plat {new_plate} di slot {slot_id} tidak valid.")
                            is_valid = False
                            similarity = 0.0 # Nilai default
                            updated_plate_info = (new_plate, is_valid, similarity, bbox_coords)
                            plates_with_boxes[0] = updated_plate_info 
                    else:
                        # Jika plat sama dengan yang terakhir divalidasi, pastikan struktur tuple konsisten
                        # Ambil data validasi dari state sebelumnya jika ada
                        existing_plates = global_camera_states[slot_id].get('plates', [])
                        if existing_plates and len(existing_plates) > 0 and len(existing_plates[0]) >= 4:
                            # Gunakan data validasi yang sudah ada
                            _, is_valid, similarity, _ = existing_plates[0]
                            updated_plate_info = (new_plate, is_valid, similarity, bbox_coords)
                            plates_with_boxes[0] = updated_plate_info
                        else:
                            # Jika tidak ada data validasi sebelumnya, set default
                            updated_plate_info = (new_plate, False, 0.0, bbox_coords)
                            plates_with_boxes[0] = updated_plate_info

                    global_camera_states[slot_id]['plates'] = plates_with_boxes
                    global_camera_states[slot_id]['last_update'] = end_time
                else:
                    global_camera_states[slot_id]['plates'] = []
                
        except Empty:
            time.sleep(PROCESS_INTERVAL_SEC) 
        except Exception as e:
            logger.error(f"Worker {slot_id} error: {e}", exc_info=True)
            time.sleep(1)

def stream_reader_worker(config: Dict[str, Any]):
    slot_id = config['parking_slot_id']
    source = int(config['source']) if config['source_type'] == 'WEBCAM' and config['source'].isdigit() else config['source']
    
    cap = None
    
    def connect_camera():
        nonlocal cap
        cap = cv2.VideoCapture(source)
        time.sleep(1) 
        if not cap.isOpened():
             logger.warning(f"Gagal membuka stream untuk {slot_id} ({source}). Mencoba lagi...")
             return False
        logger.info(f"Stream {slot_id} terhubung.")
        return True

    if not connect_camera():
        with global_state_lock:
             global_camera_states[slot_id]['active'] = False
        return

    with global_state_lock:
        frame_queue: Queue = global_camera_states[slot_id]['frame_queue']

    try:
        while True:
            ret, frame = cap.read()

            resized_frame = cv2.resize(frame, (640, 480))
            
            if not ret or frame is None:
                logger.warning(f"Stream {slot_id} hilang. Mencoba koneksi ulang...")
                cap.release()
                time.sleep(1)
                if not connect_camera():
                    time.sleep(5) 
                    continue
                continue

            # 1. Simpan frame terbaru untuk display oleh Main Thread
            with global_state_lock:
                global_camera_states[slot_id]['current_frame'] = resized_frame
                
            # 2. Kirim frame terbaru untuk di-OCR (Worker Thread)
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except Empty: pass
            frame_queue.put_nowait(resized_frame.copy()) 
            
            # Kecilkan delay di sini agar I/O cepat
            time.sleep(0.001)

    except Exception as e:
        logger.critical(f"Thread {slot_id} terminated unexpectedly: {e}")
    finally:
        # Final cleanup
        frame_queue.put(None) 
        if cap:
            cap.release()
        with global_state_lock:
            global_camera_states[slot_id]['active'] = False
        logger.info(f"Stream Reader {slot_id} dihentikan.")

if __name__ == '__main__':
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        cameras = settings['cameras']
    except Exception as e:
        logger.critical(f"Gagal memuat settings.json: {e}")
        exit()
    
    threads = []
    
    for config in cameras:
        slot_id = config['parking_slot_id']
        
        # 1. Inisialisasi State Kamera
        global_camera_states[slot_id] = {
            'frame_queue': Queue(maxsize=1),
            'plates': [],
            'last_update': 0.0,
            'current_frame': None, # Frame terbaru untuk display
            'active': True,
            'config': config,
            'last_sound_time': 0.0,
            'sound_cooldown': 10.0
        }

        # 2. Buat Worker Thread (OCR)
        worker_t = threading.Thread(target=ocr_worker, args=(slot_id,), daemon=True, name=f'OCR-{slot_id}')
        threads.append(worker_t)
        
        # 3. Buat Stream Reader Thread (I/O HEADLESS)
        stream_t = threading.Thread(target=stream_reader_worker, args=(config,), daemon=True, name=f'Stream-{slot_id}')
        threads.append(stream_t)

    for t in threads:
        t.start()
    
    logger.info(f"Menjalankan {len(cameras)} kamera. Tekan 'q' di salah satu jendela untuk keluar.")

    try:
        # Loop utama yang menangani display (HARUS DI MAIN THREAD)
        while True:
            active_cameras = []
            
            with global_state_lock:
                active_cameras = [s for s_id, s in global_camera_states.items() if s['active']]

            if not active_cameras:
                break # Semua kamera non-aktif, tutup aplikasi.

            for state in active_cameras:
                slot_id = state['config']['parking_slot_id']
                frame = state['current_frame']
                sound_thread = None

                if frame is not None:
                    display_frame = frame.copy()
                    current_plates_info = state['plates']
                    last_update = state['last_update']
                    
                    # --- Drawing Logic (Hanya di Main Thread) ---
                    display_text_main = f"Slot: {slot_id} | Mencari..."
                    main_color = (0, 255, 255)
                    
                    if current_plates_info and len(current_plates_info) > 0:
                        plate_data = current_plates_info[0]
                        if len(plate_data) >= 4:
                            first_plate_string = plate_data[0]
                            is_valid = plate_data[1]
                            similarity = plate_data[2]
                        else:
                            # Fallback untuk struktur data lama
                            first_plate_string = plate_data[0]
                            is_valid = False
                            similarity = 0.0

                        current_time = time.time()
                        last_sound_time = state.get('last_sound_time', 0.0)
                        cooldown = state.get('sound_cooldown', 3.0) 

                        if current_time - last_sound_time >= cooldown:
                            
                            if is_valid == True:
                                sound_func = play_valid_sound
                            elif is_valid == False:
                                sound_func = play_invalid_sound
                            else:
                                sound_func = None

                            if sound_func:
                                sound_thread = threading.Thread(target=sound_func, daemon=True)
                                sound_thread.start()
                                
                                state['last_sound_time'] = current_time
                                logger.info(f"Suara diputar untuk {slot_id} ({'Valid' if is_valid else 'Invalid'}). Cooldown diaktifkan.")

                        time_diff = time.time() - last_update
                        display_text_main = f"Slot: {slot_id} | {first_plate_string} ({'Valid' if is_valid else 'Invalid'} {similarity:.2f}) ({time_diff:.1f}s)"
                        main_color = (0, 255, 0) if is_valid else (0, 0, 255)
                        
                        for plate_data in current_plates_info:
                            if len(plate_data) >= 4:
                                plate_string, _, _, bbox = plate_data
                                x, y, w, h = bbox
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(display_frame, plate_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                    cv2.putText(display_frame, display_text_main, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
                    cv2.imshow(f'LPR: {slot_id}', display_frame)

            # Panggilan WAJIB untuk menjaga GUI tetap responsif
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                break
            
            time.sleep(0.01) # Small delay

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Membersihkan dan menghentikan semua thread...")
        
        for state in global_camera_states.values():
            state['frame_queue'].put(None)
            
        cv2.destroyAllWindows()
        logger.info("Aplikasi Multi-Camera LPR dihentikan.")
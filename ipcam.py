import easyocr
import cv2
import numpy as np
import re
import threading
from queue import Queue, Empty
import time
import json

# --- Konfigurasi ---
SOURCE = "WEBCAM" # "WEBCAM" atau "RTSP"
RTSP_URL = 'rtsp://user:password@ip_address:port/stream_path'
WEBCAM_INDEX = 0

LANGUAGES = ['en'] 
ALLOW_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
PROCESS_INTERVAL_MS = 1000

frame_queue = Queue(maxsize=1) 

plate_result = {
    'plates': [],
    'last_update': 0.0
} 
result_lock = threading.Lock()

try:
    reader = easyocr.Reader(LANGUAGES, gpu=False) 
    print("[✅ INFO] EasyOCR Reader siap digunakan (CPU Mode).")
except Exception as e:
    print(f"[❌ ERROR] Gagal inisialisasi EasyOCR: {e}")
    exit()

def clean_license_plate(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^A-Z0-9 ]', '', text.upper()) 
    
    # Mencocokkan pola plat Indonesia dengan fleksibilitas spasi:
    # ^([A-Z]{1,2})  -> Kode Wilayah (KB)
    # \s*([0-9]{1,4}) -> Nomor Pendaftaran (\s* untuk opsional spasi) (461)
    # \s*([A-Z]{0,3})?$ -> Sufiks (L)
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

def scan_plate_simple(frame: np.ndarray) -> list[tuple[str, list]] :
    if frame is None or frame.size == 0:
        return []

    found_plates_with_boxes = []
    
    try:
        results = reader.readtext(frame, allowlist=ALLOW_LIST)
        
        # 1. Sortir hasil berdasarkan koordinat Y (dari atas ke bawah)
        # dan kemudian koordinat X (dari kiri ke kanan) untuk urutan yang benar.
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0])) 

        full_text_segments = []
        combined_bbox_points = []
        
        # 2. Kumpulkan semua segmen teks dan titik bounding box
        for (bbox_corners, text, confidence) in results:
            # Hanya kumpulkan teks yang cukup tinggi kepercayaannya (misal > 0.5)
            if confidence > 0.5 and text.strip(): 
                full_text_segments.append(text)
                combined_bbox_points.extend(bbox_corners) # Kumpulkan semua titik sudut
                
        # 3. Gabungkan semua teks yang terdeteksi menjadi satu string
        full_raw_text = ' '.join(full_text_segments)

        if not full_raw_text:
            return []

        # 4. Validasi string gabungan
        cleaned_plate = clean_license_plate(full_raw_text)

        if cleaned_plate and combined_bbox_points:
            
            # 5. Hitung bounding box gabungan (untuk display)
            x_coords = [p[0] for p in combined_bbox_points]
            y_coords = [p[1] for p in combined_bbox_points]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            
            # Kembalikan plat yang sudah bersih dan bounding box gabungan
            found_plates_with_boxes.append((cleaned_plate, [x, y, w, h]))
            
        return found_plates_with_boxes
            
    except Exception as e:
        print(f"[⚠️ ERROR] Error saat memproses frame: {e}")
        return []
    
def ocr_worker():
    """Thread yang bertanggung jawab menjalankan OCR secara periodik."""
    global plate_result
    plate_results = []

    while True:
        try:
            frame = frame_queue.get_nowait() 
            
            if frame is None: # Sentinel untuk menghentikan thread
                break

            start_time = time.time()
            
            # Jalankan proses berat (OCR)
            # Sekarang mengembalikan list of (plate_string, bbox)
            plates_with_boxes = scan_plate_simple(frame)
            
            end_time = time.time()
            
            with result_lock:
                if plates_with_boxes:
                    plate_result['plates'] = plates_with_boxes
                    plate_result['last_update'] = end_time
                # Jika tidak ada plat yang ditemukan, plate_result['plates'] akan tetap kosong
                # agar display tidak menampilkan plat yang salah
                else:
                    plate_result['plates'] = []
                
            print(f"[WORKER] OCR selesai dalam {int((end_time - start_time) * 1000)}ms. Ditemukan {len(plates_with_boxes)} plat.")
            time.sleep(PROCESS_INTERVAL_MS / 1000.0)
            
        except Empty:
            time.sleep(PROCESS_INTERVAL_MS / 1000.0) 
        except Exception as e:
            print(f"[WORKER] Error tak terduga: {e}")
            time.sleep(1)
        finally:
            if 'frame' in locals() and frame is not None:
                plate_results.append(plate_result)
                if len(plate_results) > 3:
                    plate_results = [result for result in plate_results if result['plates']]
                    if plate_results:
                        final_plate_result = max(plate_results, key=lambda x: x['last_update'])
                        print(f"[WORKER] Hasil OCR final (dari {len(plate_results)} hasil): {final_plate_result}")
                    plate_results = []
                frame_queue.task_done()

# --- Main Thread (I/O dan Display) ---

if __name__ == '__main__':
    # Inisialisasi thread OCR
    worker_thread = threading.Thread(target=ocr_worker, daemon=True)
    worker_thread.start()
    
    # Inisialisasi video capture
    if SOURCE == "WEBCAM":
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    else:
        cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print(f"[❌ ERROR] Gagal membuka stream: {'Webcam' if SOURCE == 'WEBCAM' else RTSP_URL}. Cek URL atau ID kamera.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Stream berakhir atau gagal membaca frame.")
                break

            # Kirim frame terbaru ke worker thread
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except Empty: pass
            frame_queue.put_nowait(frame.copy()) 

            # --- Ambil hasil terbaru dan gambar bounding box ---
            with result_lock:
                current_plates_info = plate_result['plates']
                last_update = plate_result['last_update']
            
            display_text_main = "Mencari Plat..."
            main_color = (0, 255, 255) # Kuning
            
            if current_plates_info:
                # Ambil plat pertama yang terdeteksi untuk tampilan utama
                first_plate_string = current_plates_info[0][0]
                time_diff = time.time() - last_update
                display_text_main = f"PLAT: {first_plate_string} (Updated {time_diff:.1f}s ago)"
                main_color = (0, 255, 0) # Hijau jika ada plat
                
                # Gambar semua bounding box yang ditemukan oleh worker
                for plate_string, bbox in current_plates_info:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
                    # Tambahkan teks plat di atas box
                    cv2.putText(frame, plate_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Tampilkan teks status utama
            cv2.putText(frame, display_text_main, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, main_color, 2)
            cv2.imshow('Real-Time LPR with Bounding Box', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Pengguna menghentikan aplikasi.")
    finally:
        print("\n[INFO] Menghentikan worker thread dan stream...")
        frame_queue.put(None) # Sinyal berhenti ke worker
        worker_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Aplikasi dihentikan.")
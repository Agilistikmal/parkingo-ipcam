import requests
import json
from gtts import gTTS
from playsound3 import playsound
import os

def validate_booking_plate(parking_slot_id: str, plate: str):
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    api_key = settings['api_key']
    api_url = settings['api_url']
    parking_id = settings['parking_id']

    payload = {
        "plate_number": plate,
        "parking_slug": parking_id,
        "slot": parking_slot_id
    }

    response = requests.post(f"{api_url}/v1/bookings/validate", json=payload, headers={
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    })

    if response.status_code != 200:
        return None
    
    response_body = response.json()
    data = response_body['data']

    return data

def play_valid_sound():
    if os.path.exists('valid.mp3'):
        playsound('valid.mp3')
    else:
        tts = gTTS(text='Parkir anda sudah di validasi', lang='id')
        tts.save('valid.mp3')
        playsound('valid.mp3')

def play_invalid_sound():
    if os.path.exists('invalid.mp3'):
        playsound('invalid.mp3')
    else:
        tts = gTTS(text=f'Slot ini telah di booking orang lain, bukan untuk kendaraan anda. Pindahkan sekarang', lang='id')
        tts.save('invalid.mp3')
        playsound('invalid.mp3')
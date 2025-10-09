import requests
import json

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


import os
import threading
import uuid
import firebase_admin
from firebase_admin import messaging, credentials
from flask import Flask, request, jsonify
from PIL import Image
from datetime import datetime, timezone
from ultralytics import YOLO
import logging

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

REGISTERED_TOKENS = set()
INVENTORY = {}

SHELF_LIFE_HOURS = {
    "banana": 72,
    "apple": 120,
    "grapes": 48,
    "pear": 50,
    "orange": 168,
    "strawberry": 48,
    "kiwi": 96,
    "pineapple": 96,
    "mango": 96,
    "blueberry": 72,
    "peach": 72,
    "plum": 96,
    "watermelon": 120,
    "lemon": 168,
    "lime": 168,
    "papaya": 96,
    "cherry": 72,
    "pomegranate": 336,
}

RESPIRATION_CONSTANTS = {
    "banana": 1.2,
    "apple": 0.7,
    "grapes": 1.0,
    "pear": 0.9,
    "orange": 0.6,
    "strawberry": 1.5,
    "kiwi": 1.1,
    "pineapple": 1.3,
    "mango": 1.2,
    "blueberry": 1.4,
    "peach": 1.1,
    "plum": 1.0,
    "watermelon": 0.6,
    "lemon": 0.5,
    "lime": 0.5,
    "papaya": 1.3,
    "cherry": 1.4,
    "pomegranate": 0.3,
}

SENSOR_DATA = {
    "temperature": 4.0,
    "humidity": 85.0,
    "ethylene_ppm": 0.0
}

Q10 = 2.0

def environment_factor_q10(temp: float, humidity: float):
    temp_factor = Q10 ** ((temp - 4.0) / 10.0)
    humidity_factor = 1.0 if humidity >= 60 else 0.85
    return temp_factor / humidity_factor

def estimate_rsl(fruit, timestamps, now_unix, temp, humidity):
    base_life = SHELF_LIFE_HOURS.get(fruit, 72)
    respiration = RESPIRATION_CONSTANTS.get(fruit, 1.0)
    env_factor = environment_factor_q10(temp, humidity)

    rsl_list = []
    for ts in timestamps:
        hours_passed = (now_unix - ts) / 3600
        adjusted_life = (base_life / respiration) / env_factor
        remaining = max(0, adjusted_life - hours_passed)
        rsl_list.append(remaining)
    return rsl_list

yolo_model = YOLO('yolov8n.pt')

cred = credentials.Certificate("/etc/secrets/fruitmonitorapp-firebase-adminsdk-fbsvc-57c4128c52.json")
firebase_admin.initialize_app(cred)

def send_fcm_alert(token: str, title: str, body: str):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        token=token
    )
    response = messaging.send(message)
    print("Sent via FCM HTTP v1:", response)

@app.route('/register-token', methods=['POST'])
def register_token():
    data = request.get_json() or {}
    token = data.get("token")
    if token:
        REGISTERED_TOKENS.add(token)
        logging.info(f"Received token: {token}")
        return jsonify({"status": "registered"}), 200
    return jsonify({"error": "no token"}), 400

@app.route('/update-sensors', methods=['POST'])
def update_sensors():
    data = request.get_json() or {}
    try:
        SENSOR_DATA["temperature"] = float(data.get("temperature", SENSOR_DATA["temperature"]))
        SENSOR_DATA["humidity"] = float(data.get("humidity", SENSOR_DATA["humidity"]))
        SENSOR_DATA["ethylene_ppm"] = float(data.get("ethylene_ppm", SENSOR_DATA["ethylene_ppm"]))
        logging.info(f"Updated sensor data: {SENSOR_DATA}")
        return jsonify({"status": "sensor data updated"}), 200
    except (TypeError, ValueError):
        return jsonify({"error": "invalid sensor values"}), 400

@app.route('/inventory', methods=['GET'])
def get_inventory():
    now = int(datetime.now(timezone.utc).timestamp())
    temp = SENSOR_DATA["temperature"]
    humidity = SENSOR_DATA["humidity"]

    inventory_with_rsl = {}
    for fruit, data in INVENTORY.items():
        # Limit to last 100 timestamps
        if len(data["timestamps"]) > 100:
            data["timestamps"] = data["timestamps"][-100:]

        rsl_list = estimate_rsl(fruit, data["timestamps"], now, temp, humidity)
        inventory_with_rsl[fruit] = {
            "timestamps": data["timestamps"],
            "rsl_hours": [round(r, 1) for r in rsl_list],
            "average_rsl": round(sum(rsl_list) / len(rsl_list), 1) if rsl_list else None,
            "min_rsl": round(min(rsl_list), 1) if rsl_list else None,
            "count": len(rsl_list)
        }

    return jsonify(inventory_with_rsl)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join("uploads", filename)
    file.save(path)

    threading.Thread(target=process_image, args=(path,)).start()

    return jsonify({'status': 'image received, processing'}), 202

def process_image(path):
    image = Image.open(path).convert('RGB')
    logging.info("Image loaded")

    now = datetime.now(timezone.utc)
    now_unix = int(now.timestamp())

    results = yolo_model.predict(image, imgsz=640, conf=0.25)
    logging.info("YOLO prediction done")
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    logging.info("Begin Processing")

    current_counts = {}
    for det in detections:
        _, _, _, _, conf, cls = det
        class_name = names[int(cls)]
        current_counts[class_name] = current_counts.get(class_name, 0) + 1

    previous_inventory = {k: list(v["timestamps"]) for k, v in INVENTORY.items()}
    alerts = []

    for fruit, count in current_counts.items():
        old_timestamps = INVENTORY.get(fruit, {}).get("timestamps", [])
        old_count = len(old_timestamps)
        is_new_class = fruit not in INVENTORY

        if is_new_class:
            INVENTORY[fruit] = {"timestamps": []}
            alerts.append(f"New item detected: {fruit}")

        if count > old_count:
            added = count - old_count
            INVENTORY[fruit]["timestamps"].extend([now_unix] * added)
            alerts.append(f"{added} more {fruit}(s) added (now {count})")

        elif count < old_count:
            removed = old_count - count
            removed_timestamps = INVENTORY[fruit]["timestamps"][:removed]
            INVENTORY[fruit]["timestamps"] = INVENTORY[fruit]["timestamps"][removed:]
            readable_times = [datetime.fromtimestamp(ts, timezone.utc).date().isoformat() for ts in removed_timestamps]
            alerts.append(f"{removed} {fruit}(s) removed — choose one of: {readable_times}")

    for fruit in list(previous_inventory.keys()):
        if fruit not in current_counts:
            alerts.append(f"All {fruit}s removed from inventory")
            INVENTORY.pop(fruit, None)

    temp = SENSOR_DATA["temperature"]
    humidity = SENSOR_DATA["humidity"]

    for fruit, data in INVENTORY.items():
        rsl_list = estimate_rsl(fruit, data["timestamps"], now_unix, temp, humidity)
        for idx, rsl in enumerate(rsl_list):
            if rsl <= 6:
                readable_ts = datetime.fromtimestamp(data["timestamps"][idx], timezone.utc).isoformat()
                alerts.append(f"RSL alert: {fruit} placed at {readable_ts} is near spoilage ({rsl:.1f}h left)")

    logging.info(alerts)
    for alert in alerts:
        for token in REGISTERED_TOKENS:
            send_fcm_alert(token, "Inventory Update", alert)

    logging.info("Processing completed. Alerts sent.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

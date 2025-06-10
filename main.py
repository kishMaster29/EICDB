import os
import threading
import uuid
import firebase_admin
from firebase_admin import messaging, credentials, firestore
from flask import Flask, request, jsonify
from PIL import Image
from datetime import datetime, timezone
from ultralytics import YOLO
import logging

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

cred = credentials.Certificate("/etc/secrets/fruitmonitorapp-firebase-adminsdk-fbsvc-57c4128c52.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

SHELF_LIFE_HOURS = {
    "banana": 72, "apple": 120, "grapes": 48, "pear": 50, "orange": 168,
    "strawberry": 48, "kiwi": 96, "pineapple": 96, "mango": 96, "blueberry": 72,
    "peach": 72, "plum": 96, "watermelon": 120, "lemon": 168, "lime": 168,
    "papaya": 96, "cherry": 72, "pomegranate": 336
}

RESPIRATION_CONSTANTS = {
    "banana": 1.2, "apple": 0.7, "grapes": 1.0, "pear": 0.9, "orange": 0.6,
    "strawberry": 1.5, "kiwi": 1.1, "pineapple": 1.3, "mango": 1.2, "blueberry": 1.4,
    "peach": 1.1, "plum": 1.0, "watermelon": 0.6, "lemon": 0.5, "lime": 0.5,
    "papaya": 1.3, "cherry": 1.4, "pomegranate": 0.3
}

SENSOR_DATA = {
    "temperature": 4.0,
    "humidity": 85.0,
    "ethylene_ppm": 0.0
}

Q10 = 2.0
yolo_model = YOLO('models/yolov8n.pt')

def environment_factor_q10(temp: float, humidity: float):
    temp_factor = Q10 ** ((temp - 4.0) / 10.0)
    humidity_factor = 1.0 if humidity >= 60 else 0.85
    return temp_factor / humidity_factor


def estimate_rsl(fruit, timestamps, now_unix, temp, humidity):
    base_life = SHELF_LIFE_HOURS.get(fruit, 72)
    respiration = RESPIRATION_CONSTANTS.get(fruit, 1.0)
    env_factor = environment_factor_q10(temp, humidity)
    return [max(0, (base_life / respiration) / env_factor - (now_unix - ts) / 3600) for ts in timestamps]

@app.route('/register-token', methods=['POST'])
def register_token():
    data = request.get_json() or {}
    token = data.get("token")
    if token:
        doc_ref = db.collection("app").document("tokens")
        existing = doc_ref.get().to_dict() or {}
        tokens = set(existing.get("tokens", []))
        tokens.add(token)
        doc_ref.set({"tokens": list(tokens)})
        logging.info(f"Registered token: {token}")
        return jsonify({"status": "registered"}), 200
    return jsonify({"error": "no token"}), 400

@app.route('/update-sensors', methods=['POST'])
def update_sensors():
    data = request.get_json() or {}
    try:
        SENSOR_DATA["temperature"] = float(data.get("temperature", SENSOR_DATA["temperature"]))
        SENSOR_DATA["humidity"] = float(data.get("humidity", SENSOR_DATA["humidity"]))
        SENSOR_DATA["ethylene_ppm"] = float(data.get("ethylene_ppm", SENSOR_DATA["ethylene_ppm"]))
        db.collection("sensors").document("latest").set(SENSOR_DATA)
        logging.info(f"Updated sensor data: {SENSOR_DATA}")
        return jsonify({"status": "sensor data updated"}), 200
    except (TypeError, ValueError):
        return jsonify({"error": "invalid sensor values"}), 400


def send_fcm_alert(token: str, title: str, body: str):
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        token=token
    )
    response = messaging.send(message)
    logging.info("Sent via FCM HTTP v1:", response)

@app.route('/inventory', methods=['GET'])
def get_inventory():
    doc = db.collection("inventory").document("current").get()
    data = doc.to_dict() or {}

    logs_ref = db.collection("inventory_logs")
    logs_docs = logs_ref.stream()

    now = int(datetime.now(timezone.utc).timestamp())
    temp = SENSOR_DATA["temperature"]
    humidity = SENSOR_DATA["humidity"]

    response = {}

    for fruit, entry in data.items():
        timestamps = entry.get("timestamps", [])[-100:]
        rsl_list = estimate_rsl(fruit, timestamps, now, temp, humidity)
        response[fruit] = {
            "timestamps": timestamps,
            "rsl_hours": [round(r, 1) for r in rsl_list],
            "average_rsl": round(sum(rsl_list) / len(rsl_list), 1) if rsl_list else None,
            "min_rsl": round(min(rsl_list), 1) if rsl_list else None,
            "count": len(rsl_list)
        }

    logs_data = {}
    for log_doc in logs_docs:
        logs_data[log_doc.id] = log_doc.to_dict().get("log", [])

    return jsonify({"current": response, "logs": logs_data})

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
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    current_counts = {}
    for det in detections:
        _, _, _, _, conf, cls = det
        class_name = names[int(cls)]
        current_counts[class_name] = current_counts.get(class_name, 0) + 1

    doc_ref = db.collection("inventory").document("current")
    inventory_data = doc_ref.get().to_dict() or {}
    alerts = []

    for fruit, count in current_counts.items():
        timestamps = inventory_data.get(fruit, {}).get("timestamps", [])
        old_count = len(timestamps)
        is_new = fruit not in inventory_data

        if is_new:
            inventory_data[fruit] = {"timestamps": []}
            alerts.append(f"New item detected: {fruit}")

        if count > old_count:
            added = count - old_count
            inventory_data[fruit]["timestamps"].extend([now_unix] * added)
            alerts.append(f"{added} more {fruit}(s) added (now {count})")

        elif count < old_count:
            removed = old_count - count
            removed_ts = inventory_data[fruit]["timestamps"][:removed]
            inventory_data[fruit]["timestamps"] = inventory_data[fruit]["timestamps"][removed:]
            readable = [datetime.fromtimestamp(ts, timezone.utc).date().isoformat() for ts in removed_ts]
            alerts.append(f"{removed} {fruit}(s) removed — choose one of: {readable}")

    for fruit in list(inventory_data.keys()):
        if fruit not in current_counts:
            alerts.append(f"All {fruit}s removed from inventory")
            inventory_data.pop(fruit, None)

    doc_ref.set(inventory_data)

    # log RSL to /inventory_logs/{fruit}
    temp = SENSOR_DATA["temperature"]
    humidity = SENSOR_DATA["humidity"]

    for fruit, data in inventory_data.items():
        timestamps = data.get("timestamps", [])
        rsl_list = estimate_rsl(fruit, timestamps, now_unix, temp, humidity)

        if rsl_list:
            avg_rsl = round(sum(rsl_list) / len(rsl_list), 1)
            min_rsl = round(min(rsl_list), 1)
            log_entry = {
                "timestamp": now_unix,
                "rsl_values": [round(r, 1) for r in rsl_list],
                "average_rsl": avg_rsl,
                "min_rsl": min_rsl
            }

            log_doc = db.collection("inventory_logs").document(fruit)
            old_logs = log_doc.get().to_dict() or {}
            logs = old_logs.get("log", [])
            logs.append(log_entry)
            if len(logs) > 30:
                logs = logs[-30:]
            log_doc.set({"log": logs})

        for idx, rsl in enumerate(rsl_list):
            if rsl <= 6:
                readable_ts = datetime.fromtimestamp(timestamps[idx], timezone.utc).isoformat()
                alerts.append(f"RSL alert: {fruit} placed at {readable_ts} is near spoilage ({rsl:.1f}h left)")

    token_doc = db.collection("app").document("tokens").get()
    tokens = token_doc.to_dict().get("tokens", []) if token_doc.exists else []

    logging.info(tokens)

    for alert in alerts:
        for token in tokens:
            send_fcm_alert(token, "Inventory Update", alert)

    logging.info("Processing completed. Alerts sent.")

@app.route('/')
def home():
    return '✅ Fruit Monitor Server is running!'

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Binding to port {port}")
    app.run(host='0.0.0.0', port=port)

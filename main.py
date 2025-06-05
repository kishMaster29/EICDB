import firebase_admin
from firebase_admin import messaging, credentials
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

REGISTERED_TOKENS = set()

INVENTORY = {}

SHELF_LIFE_HOURS = {
    "banana": 72,
    "apple": 120,
    "grapes": 48,
    "pear": 50,
}

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
        return jsonify({"status": "registered"}), 200
    return jsonify({"error": "no token"}), 400


@app.route('/inventory', methods=['GET'])
def get_inventory():
    return jsonify(INVENTORY)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(BytesIO(file.read())).convert('RGB')

    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    # YOLO detection
    results = yolo_model.predict(image, imgsz=640, conf=0.25)
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    # Analyze detections
    current_counts = {}
    for det in detections:
        _, _, _, _, conf, cls = det
        class_name = names[int(cls)]
        current_counts[class_name] = current_counts.get(class_name, 0) + 1

    previous_inventory = {k: list(v["timestamps"]) for k, v in INVENTORY.items()}
    alerts = []

    # Check new detections vs old
    for fruit, count in current_counts.items():
        old_timestamps = INVENTORY.get(fruit, {}).get("timestamps", [])
        old_count = len(old_timestamps)
        is_new_class = fruit not in INVENTORY

        if is_new_class:
            INVENTORY[fruit] = {"timestamps": []}
            alerts.append(f"New item detected: {fruit}")

        if count > old_count:
            added = count - old_count
            INVENTORY[fruit]["timestamps"].extend([now_iso] * added)
            alerts.append(f"{added} more {fruit}(s) added (now {count})")

        elif count < old_count:
            removed = old_count - count
            removed_timestamps = INVENTORY[fruit]["timestamps"][:removed]
            INVENTORY[fruit]["timestamps"] = INVENTORY[fruit]["timestamps"][removed:]
            alerts.append(f"{removed} {fruit}(s) removed — choose one of: {removed_timestamps}")

    # Check for completely removed items
    for fruit in list(previous_inventory.keys()):
        if fruit not in current_counts:
            alerts.append(f"All {fruit}s removed from inventory")
            INVENTORY.pop(fruit, None)

    # Spoilage check
    for fruit, data in INVENTORY.items():
        for ts in data["timestamps"]:
            ts_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            hours_passed = (now - ts_dt).total_seconds() / 3600
            shelf_life = SHELF_LIFE_HOURS.get(fruit, float('inf'))
            if hours_passed > shelf_life:
                alerts.append(f"Spoilage alert: {fruit} from {ts} (>{shelf_life}h)")

    # Send all alerts via FCM
    for alert in alerts:
        for token in REGISTERED_TOKENS:
            send_fcm_alert(token, "Inventory Update", alert)

    return jsonify({"status": "processed", "alerts_sent": alerts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

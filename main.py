from ultralytics import YOLO
import cv2
import numpy as np
import requests
import os

# ---------------- CONFIG ----------------

# Set to True to use webcam instead of video file
USE_WEBCAM = False
VIDEO_PATH = "data/test_feed.mp4"

# Local YOLO model
MODEL_NAME = "yolov8n.pt"

# Detection engine: "YOLO", "AZURE", "HYBRID"
DETECTION_MODE = "YOLO"  # change to "AZURE" or "HYBRID" later

# Azure Computer Vision config (fill these in when you create the resource)
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com")
AZURE_KEY = os.getenv("AZURE_VISION_KEY", "<your-key-here>")

# Your interesting classes (COCO IDs -> names)
INTERESTING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    7: "truck",

    # Animals (Africa-relevant via COCO)
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    22: "zebra",
    23: "giraffe",

    # Other objects you kept
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
}

# For convenience, build a set of the names we care about
INTERESTING_NAMES = set(INTERESTING_CLASSES.values())


# ---------------- RISK SCORE ----------------

def compute_risk_score(detections):
    """
    Very simple risk function:
      - people weighted highest
      - vehicles + animals contribute to risk
      - bonus risk if people and animals appear together
    """
    num_person = sum(1 for d in detections if d["cls_name"] == "person")
    num_vehicle = sum(1 for d in detections if d["cls_name"] in ["car", "truck", "bus", "motorcycle", "bicycle", "airplane"])
    num_animal = sum(1 for d in detections if d["cls_name"] in [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    ])

    base = num_person * 3 + num_vehicle * 2 + num_animal * 2

    # if people and animals in the same frame, increase risk
    if num_person > 0 and num_animal > 0:
        base += 5

    # clip to 0-100
    return int(max(0, min(100, base)))


# ---------------- YOLO DETECTION ----------------

yolo_model = None  # lazy-load once


def detect_with_yolo(frame):
    """
    Run local YOLO (yolov8n.pt) and return a list of detection dicts.
    Each detection:
      { "cls_name": str, "conf": float, "bbox": (x1, y1, x2, y2), "source": "yolo" }
    """
    global yolo_model
    if yolo_model is None:
        print("[YOLO] Loading model...")
        yolo_model = YOLO(MODEL_NAME)

    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes

    detections = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id not in INTERESTING_CLASSES:
                continue

            cls_name = INTERESTING_CLASSES[cls_id]

            detections.append({
                "cls_name": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "source": "yolo"
            })

    return detections


# ---------------- AZURE DETECTION ----------------

def _normalize_azure_label(raw_name: str) -> str:
    """
    Map Azure object names to our internal class names as best as possible.
    Azure returns things like 'person', 'car', 'dog', etc.
    """
    name = raw_name.lower().strip()

    # direct matches
    if name in INTERESTING_NAMES:
        return name

    # simple mappings / startswith for animals & vehicles
    mapping = {
        "human": "person",
        "man": "person",
        "woman": "person",
        "boy": "person",
        "girl": "person",
        "automobile": "car",
        "vehicle": "car",
        "truck": "truck",
        "bus": "bus",
        "bicycle": "bicycle",
        "motorbike": "motorcycle",
        "motorcycle": "motorcycle",
        "plane": "airplane",
        "aeroplane": "airplane",
        "bike": "bicycle",

        # animals
        "dog": "dog",
        "cat": "cat",
        "bird": "bird",
        "horse": "horse",
        "sheep": "sheep",
        "cow": "cow",
        "elephant": "elephant",
        "zebra": "zebra",
        "giraffe": "giraffe",
    }

    for key, val in mapping.items():
        if name == key or name.startswith(key):
            return val

    # If we don't know it or don't care about it
    return ""


def detect_with_azure(frame):
    """
    Call Azure Vision object detection and return detections in same format as YOLO.
    You MUST configure AZURE_ENDPOINT and AZURE_KEY.
    """
    # encode frame as JPEG
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return []

    img_bytes = buf.tobytes()

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream",
    }

    # Example endpoint path for object detection (Image Analysis 3.x/4.x)
    # Adjust to your actual API version
    url = AZURE_ENDPOINT.rstrip("/") + "/vision/v3.2/detect"

    try:
        resp = requests.post(url, headers=headers, data=img_bytes, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[AZURE ERROR] {e}")
        return []

    detections = []

    # Azure returns "objects": [ { "object": name, "confidence": x, "rectangle": {x,y,w,h} }, ... ]
    for obj in data.get("objects", []):
        raw_name = obj.get("object", "")
        conf = float(obj.get("confidence", 0.0))
        rect = obj.get("rectangle", {})

        cls_name = _normalize_azure_label(raw_name)
        if not cls_name or cls_name not in INTERESTING_NAMES:
            continue  # ignore things we don't track

        x = rect.get("x", 0)
        y = rect.get("y", 0)
        w = rect.get("w", 0)
        h = rect.get("h", 0)
        x1, y1, x2, y2 = x, y, x + w, y + h

        detections.append({
            "cls_name": cls_name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "source": "azure"
        })

    return detections


# ---------------- ROUTER (YOLO / AZURE / HYBRID) ----------------

def get_detections(frame, frame_idx):
    """
    Route detection to YOLO, Azure, or both (HYBRID).
    """
    if DETECTION_MODE == "YOLO":
        return detect_with_yolo(frame)

    elif DETECTION_MODE == "AZURE":
        return detect_with_azure(frame)

    elif DETECTION_MODE == "HYBRID":
        # Strategy: always run YOLO, and every N frames also call Azure, then merge
        yolo_dets = detect_with_yolo(frame)

        azure_dets = []
        # e.g. every 30th frame, call Azure to show cloud integration
        if frame_idx % 30 == 0:
            azure_dets = detect_with_azure(frame)
            print(f"[HYBRID] Azure detected {len(azure_dets)} objects on frame {frame_idx}")

        # Merge (this may double-count if both see the same object, which is okay for MVP)
        return yolo_dets + azure_dets

    else:
        return []


# ---------------- MAIN LOOP ----------------

def main():
    print(f"[INFO] Detection mode: {DETECTION_MODE}")

    # Open video source
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    print("Starting detection. Press 'ESC' to quit.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        # If using the video file and we reach the end we loop
        if not ret:
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # ---- Get detections from the selected engine(s)
        detections = get_detections(frame, frame_idx)

        # ---- Compute risk score
        risk_score = compute_risk_score(detections)

        # ---- Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            source = det.get("source", "yolo")

            # default color
            color = (0, 255, 0)

            # humans -> yellow
            if cls_name == "person":
                color = (0, 255, 255)
            # "wildlife-ish" -> red
            if cls_name in ["horse", "sheep", "cow", "bird", "dog", "cat", "elephant", "bear", "zebra", "giraffe"]:
                color = (0, 0, 255)
            # vehicles -> blue
            if cls_name in ["car", "truck", "bus", "motorcycle", "bicycle", "airplane"]:
                color = (255, 0, 0)

            # slightly different border for Azure (optional visual cue)
            thickness = 2 if source == "yolo" else 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f"{cls_name} {conf:.2f} ({source})",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ---- Draw risk score banner
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        if risk_score < 30:
            risk_color = (0, 255, 0)
            risk_label = "LOW"
        elif risk_score < 70:
            risk_color = (0, 255, 255)
            risk_label = "MEDIUM"
        else:
            risk_color = (0, 0, 255)
            risk_label = "HIGH"

        text = f"Risk Score: {risk_score} ({risk_label})   |   Objects: {len(detections)}"
        cv2.putText(frame, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)

        cv2.imshow("TeraWatch - YOLO/Azure Hybrid Detection & Risk", frame)

        # ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

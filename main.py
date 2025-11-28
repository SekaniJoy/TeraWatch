from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

USE_WEBCAM = False
VIDEO_PATH = "data/test_feed.mp4"
MODEL_NAME = "yolov8m.pt"

DETECTION_MODE = "HYBRID"  # YOLO, ROBOFLOW, HYBRID (This is the default setting)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = "savanna-animals/2"
ROBOFLOW_INTERVAL = 15 # frames (Run Roboflow cloud detection every 15 frames)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DETECTION_LOG = LOG_DIR / "detections.csv"
ALERT_LOG = LOG_DIR / "alerts.csv"

# -----------------------------------------------------------------------------
# CLASS LISTS (for risk scoring)
# -----------------------------------------------------------------------------

PERSON_CLASSES = {"person", "man", "woman", "human"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane"}

WILDLIFE_CLASSES = {
    "elephant", "zebra", "giraffe", "lion", "leopard", "cheetah", 
    "rhinoceros", "rhino", "buffalo", "hippopotamus", "hippo", "hyena", 
    "ostrich", "camel", "crocodile", "gazelle", "impala", "antelope", 
    "baboon", "wildebeest", "warthog", 
}

# -----------------------------------------------------------------------------
# ZONE DEFINITIONS (frame coordinates; adjust per feed)
# -----------------------------------------------------------------------------

ZONES = [
    {
        "name": "North Border Line",
        "type": "border",
        "polygon": [(100, 80), (520, 80), (520, 140), (100, 140)],
    },
    {
        "name": "Reserve Core",
        "type": "reserve",
        "polygon": [(150, 200), (600, 200), (600, 420), (150, 420)],
    },
]

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def load_model() -> YOLO:
    print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
    return YOLO(MODEL_NAME)


def roboflow_available() -> bool:
    return bool(ROBOFLOW_API_KEY)


def determine_final_mode(initial_mode: str) -> str:
    """
    Determines the effective detection mode based on API key availability.
    This logic is moved outside of main() to avoid the SyntaxError.
    """
    if initial_mode in {"ROBOFLOW", "HYBRID"} and not roboflow_available():
        print("[WARNING] ROBOFLOW_API_KEY not found in environment. Switching to YOLO-only mode.")
        return "YOLO"
    return initial_mode


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    # Standard ray-casting algorithm to check if a point is inside a polygon
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
        ):
            inside = not inside
    return inside


def annotate_zones(detections: List[Dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        det["center"] = center
        det["zones"] = []
        for zone in ZONES:
            if point_in_polygon(center, zone["polygon"]):
                det["zones"].append(zone["name"])


def detect_with_yolo(model: YOLO, frame: np.ndarray) -> List[Dict]:
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    detections = []
    if boxes is None or len(boxes) == 0:
        return detections
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, "object")
        conf = float(box.conf[0].item())
        detections.append(
            {
                "cls_name": cls_name.lower(),
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "source": "yolo",
            }
        )
    return detections


def detect_with_roboflow(frame: np.ndarray) -> List[Dict]:
    if not roboflow_available():
        return []

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return []

    try:
        CLIENT = InferenceHTTPClient(
            api_url="https://infer.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )

        result = CLIENT.infer(
            buf.tobytes(),
            model_id=ROBOFLOW_MODEL_ID,
        )

    except Exception as e:
        print(f"[WARN] Roboflow detection skipped: {e}")
        return []

    detections = []
    for pred in result.get("predictions", []):
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

        detections.append({
            "cls_name": pred["class"].lower(),
            "conf": pred["confidence"],
            "bbox": (x1, y1, x2, y2),
            "source": "roboflow"
        })

    return detections


def merge_detections(*det_lists: List[List[Dict]]) -> List[Dict]:
    merged = []
    for dets in det_lists:
        merged.extend(dets)
    return merged


def compute_risk_score(detections: List[Dict]) -> int:
    num_people = sum(1 for d in detections if d["cls_name"] in PERSON_CLASSES)
    num_vehicles = sum(1 for d in detections if d["cls_name"] in VEHICLE_CLASSES)
    num_wildlife = sum(1 for d in detections if d["cls_name"] in WILDLIFE_CLASSES)

    # Base risk calculation (Weights are: People=4, Vehicle=3, Wildlife=2)
    base = num_people * 4 + num_vehicles * 3 + num_wildlife * 2

    # Interaction and Intrusion modifiers
    has_person = num_people > 0
    has_wildlife = num_wildlife > 0
    has_vehicle = num_vehicles > 0
    # Check if ANY detection falls into a 'border' zone
    has_border = any("border" in zone.lower() for d in detections for zone in d.get("zones", []))

    # High-Risk Interaction: Human-Wildlife Conflict
    if has_person and has_wildlife:
        base += 10 # Major risk modifier
    
    # Medium-Risk Interaction: Unauthorized Vehicle/Person in Proximity
    if has_person and has_vehicle:
        base += 5 # Indicates human presence with means of transport

    # Intrusion Risk: Any presence near the defined border
    if has_border:
        base += 5 # Indicates a perimeter breach/monitoring need

    # Scale score to 0-100 (clamp at 100)
    return int(max(0, min(100, base)))
    


def write_log(path: Path, row: List):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "camera",
                    "cls_name",
                    "confidence",
                    "zones",
                    "source",
                ]
            )
        writer.writerow(row)


def write_alert(message: str, severity: str, data: Dict):
    exists = ALERT_LOG.exists()
    with ALERT_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "severity", "message", "data"])
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                severity,
                message,
                data,
            ]
        )


def evaluate_alerts(detections: List[Dict]):
    for det in detections:
        if not det.get("zones"):
            continue
        for zone in det["zones"]:
            if "border" in zone.lower() and det["cls_name"] in PERSON_CLASSES:
                write_alert(
                    f"Human detected in {zone}",
                    "high",
                    {"class": det["cls_name"], "zone": zone},
                )
            if "reserve" in zone.lower() and det["cls_name"] in VEHICLE_CLASSES:
                write_alert(
                    f"Vehicle inside reserve zone {zone}",
                    "medium",
                    {"class": det["cls_name"], "zone": zone},
                )


def main():
    # --- FIX APPLIED: Determine mode before the main loop starts ---
    final_detection_mode = determine_final_mode(DETECTION_MODE)

    model = load_model()

    # Opening the video source
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    print(f"Starting detection in {final_detection_mode} mode. Press 'ESC' to quit.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        yolo_dets = []
        roboflow_dets = []

        # --- Detection Routing using the fixed mode ---
        if final_detection_mode in {"YOLO", "HYBRID"}:
            yolo_dets = detect_with_yolo(model, frame)
            
        if final_detection_mode in {"ROBOFLOW", "HYBRID"} and frame_idx % ROBOFLOW_INTERVAL == 0:
            roboflow_dets = detect_with_roboflow(frame)

        # Merge local and cloud detections
        detections = merge_detections(yolo_dets, roboflow_dets)
        
        # --- Post-Processing & Logging ---
        annotate_zones(detections)
        risk_score = compute_risk_score(detections)
        evaluate_alerts(detections)

        timestamp = datetime.utcnow().isoformat()
        for det in detections:
            write_log(
                DETECTION_LOG,
                [
                    timestamp,
                    "TeraWatchCam",
                    det["cls_name"],
                    f"{det['conf']:.2f}",
                    "|".join(det.get("zones", [])),
                    det["source"],
                ],
            )
            
        # --- Visualization (OpenCV) ---
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            
            # Color-coding for visualization
            color = (0, 255, 0) # Default Green
            if cls_name in PERSON_CLASSES:
                color = (0, 255, 255) # Yellow
            elif cls_name in VEHICLE_CLASSES:
                color = (255, 140, 0) # Orange/Blue
            elif cls_name in WILDLIFE_CLASSES:
                color = (0, 0, 255) # Red

            # Ensure coordinates are integers
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} {conf:.2f} ({det['source']})"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Draw Zone Polygons
        for zone in ZONES:
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 1)
            cv2.putText(
                frame,
                zone["name"],
                pts[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Risk Score Display
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        if risk_score < 30:
            risk_color = (0, 180, 0)
            risk_label = "LOW"
        elif risk_score < 70:
            risk_color = (0, 200, 255)
            risk_label = "MEDIUM"
        else:
            risk_color = (0, 0, 255)
            risk_label = "HIGH"
        summary = f"Risk {risk_score:03d} ({risk_label}) | Objects: {len(detections)} | Mode: {final_detection_mode}"
        cv2.putText(
            frame,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            risk_color,
            2,
        )

        cv2.imshow("TeraWatch Surveillance", frame)
        

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
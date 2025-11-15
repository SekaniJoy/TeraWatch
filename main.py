from ultralytics import YOLO
import cv2
import numpy as np

#Set to 0 to use webcam instead of video file
USE_WEBCAM = False
VIDEO_PATH = "data/test_feed.mp4"
MODEL_NAME = "yolov8n.pt"

INTERESTING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
}

def compute_risk_score(detections):
    # Very simple risk function for now:
    #     -person near 'wildlife' => higher risk
    #     - more objects => higher risk
    # We'll just start with counts; later we will add zones/geofences.
    num_person = sum(1 for d in detections if d["cls_name"] == "person")
    num_vehicle = sum(1 for d in detections if d["cls_name"] in ["car", "truck", "bus", "motorcycle", "bicycle", "airplane"])
    num_animal = sum(1 for d in detections if d["cls_name"] in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"])

    base = num_person * 3 + num_vehicle * 2 + num_animal * 2

    #if people and animals in the same fram, increase risk
    if num_person > 0 and num_animal > 0:
        base += 5

    #clip to 0-100
    return int(max(0, min(100,base)))

def main():
    #Loading YOLO model
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    # Opening the video source
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    print("Starting detection. Press 'ESC' to quit.")

    while True:
        ret, frame = cap.read()

        # If using the video file and we reach the end we loop
        if not ret:
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        #run YOLO on the frame
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float (box.conf[0].item())

                if cls_id not in INTERESTING_CLASSES:
                    continue

                cls_name = INTERESTING_CLASSES[cls_id]

                detections.append({
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        # Compute a basic risk score from detections
        risk_score = compute_risk_score(detections)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]

            color = (0, 255, 0)  # default green

            # if person + animal exist in same frame, make animals red-ish
            if cls_name == "person":
                color = (0, 255, 255)  # yellow-ish for humans
            if cls_name in ["horse", "sheep", "cow", "bird", "dog", "cat"]:
                color = (0, 0, 255)    # red-ish for "wildlife"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw risk score banner at top
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        # Color of text based on risk
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

        cv2.imshow("SavannaGuard AI - Detection & Risk", frame)

        # ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import numpy as np

# Modeli yükle
model = YOLO("best.pt")  # best.pt dosyası aynı klasörde olmalı

# Kamera başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    results = model.predict(source=frame, conf=0.4, verbose=False)
    detections = results[0].boxes

    blurred = cv2.GaussianBlur(original, (45, 45), 0)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    kutular = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if (x2 - x1) * (y2 - y1) < 1000:
            continue

        kutular.append((x1, y1, x2, y2, label, conf))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask_3ch = cv2.merge([mask, mask, mask])
    final = np.where(mask_3ch == 255, original, blurred)

    for x1, y1, x2, y2, label, conf in kutular:
        cv2.rectangle(final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(final, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Smart Product Detection", final)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

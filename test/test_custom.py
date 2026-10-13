import cv2
import sys
from pathlib import Path

# Cho phép import config.py từ thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from config import CONF_THRESHOLD, load_model

# ==========================================
# LOAD MODEL DÙNG CHUNG
# ==========================================
model = load_model()

# ==========================================
# MỞ WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

print("================================")
print(" CHECKSLEEPONE - YOLO WEBCAM")
print("================================")
print(f"Confidence: {CONF_THRESHOLD:.2f}")
print("ESC = thoát")
print()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    # YOLO nhận diện
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # In thông tin nhận diện
    for box in results[0].boxes:
        print(
            f"Class: {int(box.cls[0])} | "
            f"Confidence: {float(box.conf[0]):.3f} | "
            f"Box: {box.xyxy[0].tolist()}"
        )

    # Vẽ kết quả
    annotated_frame = results[0].plot()
    cv2.imshow("CheckSleepOne YOLO", annotated_frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

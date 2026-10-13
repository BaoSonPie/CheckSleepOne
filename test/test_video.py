from pathlib import Path

import cv2
import sys
from pathlib import Path

# Cho phép import config.py từ thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from config import BASE_DIR, CONF_THRESHOLD, load_model

# ==========================================
# LOAD MODEL DÙNG CHUNG
# ==========================================
model = load_model()

# ==========================================
# VIDEO INPUT
# ==========================================
video_path = Path(r"D:\bctn\video\2026_0916_150745_560B.MP4")


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Không mở được video: {video_path}")
    exit()

# ==========================================
# VIDEO INFORMATION
# ==========================================
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("================================")
print(" CHECKSLEEPONE - YOLO VIDEO")
print("================================")
print(f"Video: {video_path}")
print(f"FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Confidence: {CONF_THRESHOLD:.2f}")
print()

# ==========================================
# OUTPUT VIDEO
# ==========================================
output_dir = BASE_DIR / "runs" / "video_test"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "cabin_yolo_v4.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height),
)

# ==========================================
# PROCESS VIDEO
# ==========================================
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO nhận diện
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Vẽ bounding box
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("CheckSleepOne - YOLO", annotated_frame)

    # Ghi video kết quả
    out.write(annotated_frame)

    # ESC để dừng
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================
# RELEASE
# ==========================================
cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("================================")
print(" TEST HOAN TAT")
print("================================")
print(f"So frame: {frame_count}")
print(f"Video ket qua: {output_path}")

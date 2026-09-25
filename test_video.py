from ultralytics import YOLO
import cv2
import os

# ==============================
# 1. LOAD MODEL V4
# ==============================
model = YOLO("runs/detect/checksleepone_eye_v4/weights/best.pt")

# ==============================
# 2. VIDEO INPUT
# ==============================
video_path = "video/2026_0916_150745_560B.MP4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không mở được video:", video_path)
    exit()

# ==============================
# 3. VIDEO INFORMATION
# ==============================
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Resolution:", width, "x", height)

# ==============================
# 4. OUTPUT VIDEO
# ==============================
os.makedirs("runs/video_test", exist_ok=True)

output_path = "runs/video_test/cabin_yolo_v4.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ==============================
# 5. PROCESS VIDEO
# ==============================
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO nhận diện
    results = model(frame, conf=0.50, verbose=False)

    # Vẽ bounding box
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("CheckSleepOne - YOLO V4", annotated_frame)

    # Ghi video
    out.write(annotated_frame)

    # ESC để dừng
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==============================
# 6. RELEASE
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("================================")
print("        TEST HOAN TAT")
print("================================")
print("So frame:", frame_count)
print("Video ket qua:", output_path)

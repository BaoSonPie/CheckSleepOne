from ultralytics import YOLO
import cv2

# Load model YOLO đã được train sẵn
model = YOLO("yolo26n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    # YOLO nhận diện frame
    results = model(frame)
    for box in results[0].boxes:
        print("Class ID:", int(box.cls[0]))
        print("Confidence:", float(box.conf[0]))
        print("Box:", box.xyxy[0].tolist())
    # Vẽ bounding box + tên vật thể + confidence
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Test", annotated_frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

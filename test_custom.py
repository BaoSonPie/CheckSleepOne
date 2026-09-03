from ultralytics import YOLO
import cv2

# Load model đã train
model = YOLO("runs/detect/checksleepone_eye/weights/best.pt")


# Mở webcam
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    # Cho model đã train nhận diện
    results = model(frame, conf=0.01)
    for box in results[0].boxes:

        print("Class:", int(box.cls[0]))

        print("Confidence:", float(box.conf[0]))

        print("Box:", box.xyxy[0].tolist())
    # Vẽ kết quả
    annotated_frame = results[0].plot()

    cv2.imshow("CheckSleepOne YOLO", annotated_frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO

model = YOLO("runs/detect/checksleepone_eye_v2-2/weights/best.pt")

results = model("dataset/images/train/20260903_182228_100162.jpg", conf=0.05)

results[0].show()

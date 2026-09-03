from ultralytics import YOLO

model = YOLO("yolo26n.pt")


model.train(data="data.yaml", epochs=20, imgsz=640, batch=8, name="checksleepone_eye")

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from config import load_model, CONF_THRESHOLD

# ==========================================
# CHECKSLEEPONE API
# ==========================================
app = FastAPI(title="CheckSleepOne API")

# Sử dụng cấu hình có sẵn trong config.py
model = load_model()


@app.get("/")
def home():
    return {"app": "CheckSleepOne", "status": "running"}


# ==========================================
# NHẬN DIỆN TRẠNG THÁI MẮT
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Kiểm tra định dạng ảnh
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400, detail="Chỉ chấp nhận ảnh JPEG, PNG hoặc WEBP"
        )

    # Đọc dữ liệu ảnh
    data = await file.read()

    if not data or len(data) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail="Ảnh rỗng hoặc vượt quá giới hạn 8 MB"
        )

    # Chuyển dữ liệu thành ảnh OpenCV
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Không thể đọc ảnh")

    # Chạy YOLO
    result = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]

    # Lấy kết quả nhận diện
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])

        detections.append(
            {
                "class_id": class_id,
                "state": result.names[class_id],
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            }
        )

    return {
        "detections": detections,
        "count": len(detections),
        "confidence_threshold": CONF_THRESHOLD,
    }

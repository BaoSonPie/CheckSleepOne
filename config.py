from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CHECKSLEEPONE - CẤU HÌNH DÙNG CHUNG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# Model YOLO dùng chung cho các chương trình kiểm thử
MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "checksleepone_eye_v4" / "weights" / "best.pt"
)

# Ngưỡng confidence dùng chung khi kiểm thử
CONF_THRESHOLD = 0.50


def load_model():
    """Load model YOLO từ cấu hình dùng chung."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model:\n{MODEL_PATH}\n"
            "Hãy kiểm tra lại đường dẫn MODEL_PATH trong config.py."
        )

    print(f"Đang load model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("Load model thành công!")
    return model

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
# ẢNH KIỂM THỬ
# ==========================================
image_path = "dataset/images/train/20260903_182228_100162.jpg"
print("================================")
print(" CHECKSLEEPONE - TEST IMAGE")
print("================================")
print(f"Image: {image_path}")
print(f"Confidence: {CONF_THRESHOLD:.2f}")
print()

# YOLO nhận diện
results = model(image_path, conf=CONF_THRESHOLD, verbose=False)
results[0].show()

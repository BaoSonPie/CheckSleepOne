from ultralytics import YOLO
import os
import shutil

# =========================================================
# CHECKSLEEPONE - AUTO CLASSIFY V4
# =========================================================

BASE_DIR = r"D:\bctn\CheckSleepOne"

# ---------------------------------------------------------
# ẢNH NGUỒN
# ---------------------------------------------------------

INPUT_DIR = os.path.join(BASE_DIR, "dataset", "video", "frame", "2026_0916_151046_562B")

# ---------------------------------------------------------
# MODEL V4
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "checksleepone_eye_v4", "weights", "best.pt"
)

# ---------------------------------------------------------
# THƯ MỤC ĐÍCH
# ---------------------------------------------------------

OPEN_DIR = os.path.join(BASE_DIR, "dataset", "raw", "open")

CLOSED_DIR = os.path.join(BASE_DIR, "dataset", "raw", "closed")

# Tạo thư mục nếu chưa có
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

# ---------------------------------------------------------
# CONFIDENCE
# ---------------------------------------------------------

CONF_THRESHOLD = 0.85

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("=" * 60)
print(" CHECKSLEEPONE - AUTO CLASSIFY V4")
print("=" * 60)

print("\nĐang load model V4...")

model = YOLO(MODEL_PATH)

print("Load model thành công!")

# ---------------------------------------------------------
# LẤY DANH SÁCH ẢNH
# ---------------------------------------------------------

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]

images.sort()

total = len(images)

print(f"\nTìm thấy {total} ảnh.")
print()

# ---------------------------------------------------------
# THỐNG KÊ
# ---------------------------------------------------------

open_count = 0
closed_count = 0
unknown_count = 0

# ---------------------------------------------------------
# XỬ LÝ
# ---------------------------------------------------------

for index, filename in enumerate(images, start=1):

    image_path = os.path.join(INPUT_DIR, filename)

    print(f"[{index}/{total}] {filename}", end=" ")

    # YOLO nhận diện
    results = model(image_path, conf=CONF_THRESHOLD, verbose=False)

    result = results[0]

    boxes = result.boxes

    # -----------------------------------------------------
    # KHÔNG PHÁT HIỆN
    # -----------------------------------------------------

    if boxes is None or len(boxes) == 0:

        print("-> KHONG NHAN DIEN")

        unknown_count += 1

        continue

    # -----------------------------------------------------
    # LẤY BOX CÓ CONFIDENCE CAO NHẤT
    # -----------------------------------------------------

    best_box = None
    best_conf = 0

    for box in boxes:

        conf = float(box.conf[0])

        if conf > best_conf:

            best_conf = conf
            best_box = box

    # -----------------------------------------------------
    # CLASS
    # -----------------------------------------------------

    class_id = int(best_box.cls[0])

    # -----------------------------------------------------
    # OPEN
    # -----------------------------------------------------

    if class_id == 0:

        destination = OPEN_DIR

        open_count += 1

        print(f"-> OPEN " f"(confidence={best_conf:.3f})")

    # -----------------------------------------------------
    # CLOSED
    # -----------------------------------------------------

    elif class_id == 1:

        destination = CLOSED_DIR

        closed_count += 1

        print(f"-> CLOSED " f"(confidence={best_conf:.3f})")

    # -----------------------------------------------------
    # CLASS KHÁC
    # -----------------------------------------------------

    else:

        print(f"-> CLASS KHONG XAC DINH: {class_id}")

        unknown_count += 1

        continue

    # -----------------------------------------------------
    # COPY ẢNH
    # -----------------------------------------------------

    shutil.copy2(image_path, os.path.join(destination, filename))

# =========================================================
# KẾT QUẢ
# =========================================================

print()
print("=" * 60)
print(" HOÀN TẤT")
print("=" * 60)

print(f"Tổng ảnh       : {total}")
print(f"Mắt MỞ         : {open_count}")
print(f"Mắt NHẮM       : {closed_count}")
print(f"Không nhận diện: {unknown_count}")

print()
print("Ảnh mắt mở:")
print(OPEN_DIR)

print()
print("Ảnh mắt nhắm:")
print(CLOSED_DIR)

print("=" * 60)

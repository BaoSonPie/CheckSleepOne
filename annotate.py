# gán nhãn

import cv2
import os
import glob
import random
import shutil

# ==========================================
# CẤU HÌNH
# ==========================================

RAW_DIR = "dataset/raw"

IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

TRAIN_RATIO = 0.8

# ==========================================
# TẠO THƯ MỤC
# ==========================================

for folder in [
    f"{IMAGE_DIR}/train",
    f"{IMAGE_DIR}/val",
    f"{LABEL_DIR}/train",
    f"{LABEL_DIR}/val",
]:
    os.makedirs(folder, exist_ok=True)


# ==========================================
# KIỂM TRA ẢNH ĐÃ ANNOTATION
# ==========================================


def is_already_annotated(filename):
    """
    Kiểm tra ảnh đã có label YOLO trong dataset hay chưa.
    """

    name = os.path.splitext(filename)[0]

    train_label = os.path.join(LABEL_DIR, "train", name + ".txt")

    val_label = os.path.join(LABEL_DIR, "val", name + ".txt")

    return os.path.exists(train_label) or os.path.exists(val_label)


# ==========================================
# LẤY DANH SÁCH ẢNH
# ==========================================

images = []

for class_name, class_id in [("open", 0), ("closed", 1)]:

    folder = os.path.join(RAW_DIR, class_name)

    files = glob.glob(os.path.join(folder, "*.jpg"))

    for file in files:

        filename = os.path.basename(file)

        # ==================================
        # BỎ QUA ẢNH ĐÃ ANNOTATION
        # ==================================

        if is_already_annotated(filename):
            continue

        images.append({"path": file, "class_id": class_id})


# ==========================================
# TRỘN ẢNH
# ==========================================

random.shuffle(images)


# ==========================================
# THÔNG BÁO
# ==========================================

print()
print("==========================================")
print("       CHECKSLEEPONE YOLO ANNOTATION")
print("==========================================")
print()

print(f"Ảnh chưa annotation: {len(images)}")

print()

if len(images) == 0:

    print("Không có ảnh mới cần annotation.")
    print("Tất cả ảnh trong dataset/raw đã được xử lý.")

    input("\nNhấn Enter để thoát...")
    exit()


# ==========================================
# BIẾN ANNOTATION
# ==========================================

current_boxes = []

drawing = False

start_x = 0
start_y = 0


# ==========================================
# MOUSE CALLBACK
# ==========================================


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global current_boxes

    # ======================================
    # BẮT ĐẦU VẼ
    # ======================================

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

    # ======================================
    # KẾT THÚC VẼ
    # ======================================

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        end_x = x
        end_y = y

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)

        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        # ==================================
        # TRÁNH BOX QUÁ NHỎ
        # ==================================

        if (x2 - x1) > 5 and (y2 - y1) > 5:

            current_boxes.append((x1, y1, x2, y2))


# ==========================================
# CỬA SỔ
# ==========================================

window_name = "YOLO Annotation"

cv2.namedWindow(window_name)

cv2.setMouseCallback(window_name, mouse_callback)


# ==========================================
# ANNOTATION LOOP
# ==========================================

index = 0

while index < len(images):

    item = images[index]

    image_path = item["path"]

    class_id = item["class_id"]

    image = cv2.imread(image_path)

    # ======================================
    # KIỂM TRA ẢNH
    # ======================================

    if image is None:

        print("Khong doc duoc:", image_path)

        index += 1

        continue

    # ======================================
    # RESET BOX
    # ======================================

    current_boxes = []

    # ======================================
    # VÒNG LẶP ẢNH
    # ======================================

    while True:

        display = image.copy()

        # ==================================
        # VẼ CÁC BOX
        # ==================================

        for box in current_boxes:

            x1, y1, x2, y2 = box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ==================================
        # THÔNG TIN
        # ==================================

        text = (
            f"{index + 1}/{len(images)} "
            f"| CLASS: "
            f"{'OPEN' if class_id == 0 else 'CLOSED'} "
            f"| BOXES: {len(current_boxes)}"
        )

        cv2.putText(
            display,
            text,
            (10, display.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "Drag mouse = box | S = save | R = reset | Q = quit",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        # ==================================
        # HIỂN THỊ ẢNH
        # ==================================

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ==================================
        # SAVE
        # ==================================

        if key == ord("s"):

            # --------------------------------
            # PHẢI CÓ BOX
            # --------------------------------

            if len(current_boxes) == 0:

                print()
                print("Chua co bounding box!")

                continue

            # --------------------------------
            # CHIA TRAIN / VAL
            # --------------------------------

            # Dùng thứ tự hiện tại để chia.
            # 80% train
            # 20% val

            if index < int(len(images) * TRAIN_RATIO):

                split = "train"

            else:

                split = "val"

            # --------------------------------
            # TÊN FILE
            # --------------------------------

            filename = os.path.basename(image_path)

            name = os.path.splitext(filename)[0]

            # --------------------------------
            # ĐƯỜNG DẪN OUTPUT
            # --------------------------------

            output_image = os.path.join(IMAGE_DIR, split, filename)

            output_label = os.path.join(LABEL_DIR, split, name + ".txt")

            # --------------------------------
            # COPY ẢNH
            # --------------------------------

            shutil.copy2(image_path, output_image)

            # --------------------------------
            # TẠO YOLO LABEL
            # --------------------------------

            height, width = image.shape[:2]

            with open(output_label, "w") as f:

                for box in current_boxes:

                    x1, y1, x2, y2 = box

                    # =========================
                    # YOLO FORMAT
                    # =========================

                    x_center = ((x1 + x2) / 2) / width

                    y_center = ((y1 + y2) / 2) / height

                    box_width = (x2 - x1) / width

                    box_height = (y2 - y1) / height

                    f.write(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{box_width:.6f} "
                        f"{box_height:.6f}\n"
                    )

            # --------------------------------
            # THÔNG BÁO
            # --------------------------------

            print(f"Saved: {filename} -> {split}")

            # --------------------------------
            # SANG ẢNH TIẾP
            # --------------------------------

            index += 1

            break

        # ==================================
        # RESET
        # ==================================

        elif key == ord("r"):

            current_boxes = []

            print("Reset boxes")

        # ==================================
        # QUIT
        # ==================================

        elif key == ord("q") or key == 27:

            cv2.destroyAllWindows()

            print()
            print("Đã thoát annotation.")

            print(f"Còn {len(images) - index} ảnh chưa xử lý.")

            exit()


# ==========================================
# KẾT THÚC
# ==========================================

cv2.destroyAllWindows()

print()
print("==========================================")
print("          ANNOTATION HOAN TAT")
print("==========================================")

print(f"Đã annotation: {len(images)} ảnh mới.")

print("Các ảnh cũ đã được tự động bỏ qua.")

print()

import cv2
import os
import glob
import random

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
# LẤY DANH SÁCH ẢNH
# ==========================================

images = []

for class_name, class_id in [("open", 0), ("closed", 1)]:

    files = glob.glob(os.path.join(RAW_DIR, class_name, "*.jpg"))

    for file in files:

        images.append({"path": file, "class_id": class_id})


random.shuffle(images)


# ==========================================
# BIẾN ANNOTATION
# ==========================================

current_boxes = []
drawing = False
start_x = 0
start_y = 0


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        end_x = x
        end_y = y

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)

        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        # Tránh box quá nhỏ
        if (x2 - x1) > 5 and (y2 - y1) > 5:

            current_boxes.append((x1, y1, x2, y2))


# ==========================================
# ANNOTATION LOOP
# ==========================================

window_name = "YOLO Annotation"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


index = 0

while index < len(images):

    item = images[index]

    image_path = item["path"]
    class_id = item["class_id"]

    image = cv2.imread(image_path)

    if image is None:

        print("Khong doc duoc:", image_path)

        index += 1
        continue

    current_boxes = []

    while True:

        display = image.copy()

        # -----------------------------
        # Vẽ các box hiện tại
        # -----------------------------

        for box in current_boxes:

            x1, y1, x2, y2 = box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # -----------------------------
        # Thông tin
        # -----------------------------

        text = (
            f"{index + 1}/{len(images)} "
            f"| CLASS: "
            f"{'OPEN' if class_id == 0 else 'CLOSED'} "
            f"| BOXES: {len(current_boxes)}"
        )

        cv2.putText(
            display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        cv2.putText(
            display,
            "Drag mouse = box | S = save | R = reset | Q = quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # -----------------------------
        # SAVE
        # -----------------------------

        if key == ord("s"):

            if len(current_boxes) == 0:

                print("Chua co bounding box!")

                continue

            # Chia train / val
            if index < int(len(images) * TRAIN_RATIO):
                split = "train"
            else:
                split = "val"

            filename = os.path.basename(image_path)

            name = os.path.splitext(filename)[0]

            # Copy image
            output_image = os.path.join(IMAGE_DIR, split, filename)

            cv2.imwrite(output_image, image)

            # Tạo label YOLO
            output_label = os.path.join(LABEL_DIR, split, name + ".txt")

            height, width = image.shape[:2]

            with open(output_label, "w") as f:

                for box in current_boxes:

                    x1, y1, x2, y2 = box

                    # YOLO format
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

            print(f"Saved: {filename} -> {split}")

            index += 1

            break

        # -----------------------------
        # RESET
        # -----------------------------

        elif key == ord("r"):

            current_boxes = []

            print("Reset boxes")

        # -----------------------------
        # QUIT
        # -----------------------------

        elif key == ord("q") or key == 27:

            cv2.destroyAllWindows()
            exit()


cv2.destroyAllWindows()

print()
print("==============================")
print("ANNOTATION HOAN TAT")
print("==============================")

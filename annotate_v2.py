import cv2
import os
import glob
import random
import shutil

# =========================================================
# CHECKSLEEPONE - ANNOTATE V2
# Hiển thị ảnh ngang/dọc vừa cửa sổ; nhãn lưu theo ảnh gốc.
# =========================================================

RAW_DIR = "dataset/raw"
IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

TRAIN_RATIO = 0.8
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 750

# Phím tắt:
# Kéo chuột trái: vẽ bounding box
# S: lưu ảnh và nhãn, sang ảnh tiếp theo
# R: xóa các box hiện tại
# U: xóa box vừa vẽ
# Q hoặc ESC: thoát

current_boxes = []
drawing = False
start_x = 0
start_y = 0
display_scale_x = 1.0
display_scale_y = 1.0
current_image = None


def ensure_directories():
    for folder in (
        os.path.join(IMAGE_DIR, "train"),
        os.path.join(IMAGE_DIR, "val"),
        os.path.join(LABEL_DIR, "train"),
        os.path.join(LABEL_DIR, "val"),
    ):
        os.makedirs(folder, exist_ok=True)


def is_already_annotated(filename):
    """Bỏ qua ảnh nếu đã có file nhãn trong train hoặc val."""
    stem = os.path.splitext(filename)[0]
    return (
        os.path.exists(os.path.join(LABEL_DIR, "train", stem + ".txt"))
        or os.path.exists(os.path.join(LABEL_DIR, "val", stem + ".txt"))
    )


def collect_images():
    """Lấy ảnh JPG/JPEG/PNG từ dataset/raw/open và dataset/raw/closed."""
    result = []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    for class_name, class_id in (("open", 0), ("closed", 1)):
        folder = os.path.join(RAW_DIR, class_name)
        files = []
        for pattern in extensions:
            files.extend(glob.glob(os.path.join(folder, pattern)))

        for path in sorted(set(files)):
            filename = os.path.basename(path)
            if not is_already_annotated(filename):
                result.append({"path": path, "class_id": class_id})

    random.shuffle(result)
    return result


def mouse_callback(event, x, y, flags, param):
    """Chuyển tọa độ trên ảnh hiển thị về tọa độ ảnh gốc."""
    global drawing, start_x, start_y, current_boxes

    if current_image is None:
        return

    h, w = current_image.shape[:2]
    original_x = max(0, min(int(round(x / display_scale_x)), w - 1))
    original_y = max(0, min(int(round(y / display_scale_y)), h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = original_x, original_y

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_x, end_y = original_x, original_y

        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))

        if (x2 - x1) > 5 and (y2 - y1) > 5:
            current_boxes.append((x1, y1, x2, y2))


def make_display(image, index, total, class_id, boxes):
    """Tạo ảnh hiển thị đã resize; boxes vẫn lưu tọa độ ảnh gốc."""
    global display_scale_x, display_scale_y

    h, w = image.shape[:2]
    scale = min(
        MAX_DISPLAY_WIDTH / w,
        MAX_DISPLAY_HEIGHT / h,
        1.0,
    )

    display_w = max(1, int(round(w * scale)))
    display_h = max(1, int(round(h * scale)))

    # Dùng tỷ lệ thực tế sau khi làm tròn kích thước.
    display_scale_x = display_w / w
    display_scale_y = display_h / h

    if (display_w, display_h) == (w, h):
        display = image.copy()
    else:
        display = cv2.resize(
            image, (display_w, display_h), interpolation=cv2.INTER_AREA
        )

    for x1, y1, x2, y2 in boxes:
        p1 = (int(round(x1 * display_scale_x)),
              int(round(y1 * display_scale_y)))
        p2 = (int(round(x2 * display_scale_x)),
              int(round(y2 * display_scale_y)))
        cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

    label = "OPEN" if class_id == 0 else "CLOSED"
    cv2.putText(
        display,
        f"{index + 1}/{total} | CLASS: {label} | BOXES: {len(boxes)}",
        (10, max(25, display_h - 40)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    cv2.putText(
        display,
        "Drag=box | S=save | R=reset | U=undo | Q=quit",
        (10, max(45, display_h - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1,
    )
    return display


def save_annotation(item, boxes, split):
    """Copy ảnh gốc và ghi nhãn YOLO chuẩn hóa theo kích thước ảnh gốc."""
    image_path = item["path"]
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]

    output_image = os.path.join(IMAGE_DIR, split, filename)
    output_label = os.path.join(LABEL_DIR, split, stem + ".txt")

    # Tránh ghi đè file cùng tên đã tồn tại ở nơi khác.
    other_split = "val" if split == "train" else "train"
    other_label = os.path.join(LABEL_DIR, other_split, stem + ".txt")
    other_image = os.path.join(IMAGE_DIR, other_split, filename)
    if os.path.exists(other_label) or os.path.exists(other_image):
        print(f"BO QUA de tranh ghi de file da ton tai: {filename}")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"Khong doc duoc anh: {image_path}")
        return False

    height, width = image.shape[:2]
    class_id = item["class_id"]

    # Ghi file tạm rồi thay thế để tránh để lại nhãn dở dang.
    temp_label = output_label + ".tmp"
    with open(temp_label, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            f.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n"
            )

    shutil.copy2(image_path, output_image)
    os.replace(temp_label, output_label)
    print(f"Saved: {filename} -> {split} ({len(boxes)} box)")
    return True


def main():
    global current_boxes, drawing, current_image

    ensure_directories()
    images = collect_images()

    print("\n==========================================")
    print("       CHECKSLEEPONE YOLO ANNOTATION V2")
    print("==========================================")
    print(f"Anh moi chua co label: {len(images)}")
    print("Class: open=0, closed=1")
    print("Luu y: moi anh can co it nhat 1 bounding box.")
    print("==========================================\n")

    if not images:
        print("Khong co anh moi can annotation.")
        input("Nhan Enter de thoat...")
        return

    window_name = "YOLO Annotation V2"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Chia ngẫu nhiên một lần cho phiên này; tránh dựa vào index của từng lần lưu.
    train_count = int(len(images) * TRAIN_RATIO)

    index = 0
    try:
        while index < len(images):
            item = images[index]
            current_image = cv2.imread(item["path"])

            if current_image is None:
                print("Khong doc duoc:", item["path"])
                index += 1
                continue

            current_boxes = []
            drawing = False

            # Xác định split trước khi lưu; không ảnh hưởng nhãn YOLO.
            split = "train" if index < train_count else "val"

            while True:
                display = make_display(
                    current_image, index, len(images),
                    item["class_id"], current_boxes
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("s"):
                    if not current_boxes:
                        print("Chua co bounding box!")
                        continue
                    if save_annotation(item, current_boxes, split):
                        index += 1
                    break

                elif key == ord("r"):
                    current_boxes.clear()
                    print("Da xoa tat ca box tren anh hien tai.")

                elif key == ord("u"):
                    if current_boxes:
                        current_boxes.pop()
                        print("Da xoa box vua ve.")

                elif key == ord("q") or key == 27:
                    print(f"\nDa thoat. Con {len(images) - index} anh chua xu ly.")
                    return

    finally:
        current_image = None
        cv2.destroyAllWindows()

    print("\n==========================================")
    print("          ANNOTATION HOAN TAT")
    print("==========================================")
    print(f"Da xu ly het danh sach anh moi: {len(images)} anh.")


if __name__ == "__main__":
    main()

import cv2
import os
from datetime import datetime

# =========================
# CẤU HÌNH
# =========================

SAVE_DIR = "dataset/raw"

os.makedirs(f"{SAVE_DIR}/open", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/closed", exist_ok=True)


# =========================
# MỞ CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()


print("================================")
print("   CHECKSLEEPONE DATA COLLECT")
print("================================")
print("O = chụp OPEN")
print("C = chụp CLOSED")
print("ESC = thoát")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    cv2.putText(
        frame,
        "O: OPEN | C: CLOSED | ESC: EXIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------
    # OPEN
    # -------------------------

    if key == ord("o"):

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")

        path = os.path.join(SAVE_DIR, "open", filename)

        cv2.imwrite(path, frame)

        print("Saved OPEN:", path)

    # -------------------------
    # CLOSED
    # -------------------------

    elif key == ord("c"):

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")

        path = os.path.join(SAVE_DIR, "closed", filename)

        cv2.imwrite(path, frame)

        print("Saved CLOSED:", path)

    # -------------------------
    # ESC
    # -------------------------

    elif key == 27:
        break


cap.release()
cv2.destroyAllWindows()
import cv2
import os
from datetime import datetime

# =========================
# CẤU HÌNH
# =========================

SAVE_DIR = "dataset/raw"

os.makedirs(f"{SAVE_DIR}/open", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/closed", exist_ok=True)


# =========================
# MỞ CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()


print("================================")
print("   CHECKSLEEPONE DATA COLLECT")
print("================================")
print("O = chụp OPEN")
print("C = chụp CLOSED")
print("ESC = thoát")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    cv2.putText(
        frame,
        "O: OPEN | C: CLOSED | ESC: EXIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------
    # OPEN
    # -------------------------

    if key == ord("o"):

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")

        path = os.path.join(SAVE_DIR, "open", filename)

        cv2.imwrite(path, frame)

        print("Saved OPEN:", path)

    # -------------------------
    # CLOSED
    # -------------------------

    elif key == ord("c"):

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")

        path = os.path.join(SAVE_DIR, "closed", filename)

        cv2.imwrite(path, frame)

        print("Saved CLOSED:", path)

    # -------------------------
    # ESC
    # -------------------------

    elif key == 27:
        break


cap.release()
cv2.destroyAllWindows()

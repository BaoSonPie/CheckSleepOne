import cv2
import csv
import hashlib
import shutil
import numpy as np

from pathlib import Path
from collections import Counter

# ============================================================
# CHECKSLEEPONE - ADVANCED FRAME FILTER
# ============================================================

# 1. THU MUC NGUON: tat ca frame nam chung mot thu muc
INPUT_DIR = Path(r"D:\bctn\video\frame\frame_pic_video")

# 2. THU MUC KET QUA: nam ngoai thu muc nguon
OUTPUT_DIR = Path(r"D:\bctn\video\frame\filtered_frames")
REVIEW_DIR = Path(r"D:\bctn\video\frame\review_frames")

REPORT_CSV = Path(r"D:\bctn\video\frame\filter_report.csv")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------
# CAU HINH CHAT LUONG ANH
# ------------------------------------------------------------

MIN_WIDTH = 100
MIN_HEIGHT = 100

# Diem do mo Laplacian: chi danh dau anh mo, khong xoa
BLUR_THRESHOLD = 30.0

# Do sang trung binh grayscale: 0-255
DARK_THRESHOLD = 25.0
BRIGHT_THRESHOLD = 245.0

# ------------------------------------------------------------
# CAU HINH LOC FRAME GAN GIONG
# ------------------------------------------------------------

ENABLE_NEAR_DUPLICATE_FILTER = True

# dHash co 64 bit. Cang nho thi cang giong nhau.
DHASH_THRESHOLD = 4

# Chenh lech pixel trung binh toi da (0-255)
MAX_PIXEL_DIFF = 6.0

# Vung xap xi vi tri mat trong anh.
# Khong phai bo phan nhan dien mat chinh xac.
EYE_Y1 = 0.20
EYE_Y2 = 0.70
EYE_X1 = 0.10
EYE_X2 = 0.90

# Neu vung mat thay doi nhieu hon muc nay,
# giu frame de tranh bo sot trang thai OPEN/CLOSED.
MAX_EYE_DIFF_FOR_DUPLICATE = 4.0

# True: anh nghi trung duoc dua vao review, khong bi xoa.
# False: anh nghi trung bi bo khoi tap filtered.
# Anh goc luon duoc giu nguyen.
COPY_NEAR_DUPLICATES_TO_REVIEW = True

# ============================================================
# KIEM TRA DUONG DAN
# ============================================================

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Khong tim thay thu muc nguon:\n{INPUT_DIR}")

if OUTPUT_DIR.resolve() == INPUT_DIR.resolve():
    raise ValueError("Thu muc dau ra khong duoc trung thu muc nguon.")

if REVIEW_DIR.resolve() == INPUT_DIR.resolve():
    raise ValueError("Thu muc review khong duoc trung thu muc nguon.")

if OUTPUT_DIR.resolve() == REVIEW_DIR.resolve():
    raise ValueError("Thu muc filtered va review phai khac nhau.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# HAM HO TRO
# ============================================================


def sha256_file(path):
    """Hash file de phat hien trung lap chinh xac."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def dhash(image):
    """Tao difference hash 64 bit."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)

    bits = small[:, 1:] > small[:, :-1]

    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return value


def hash_distance(a, b):
    """So bit khac nhau giua hai dHash."""
    return (a ^ b).bit_count()


def resize_gray(image, size=(320, 240)):
    """Chuan hoa kich thuoc truoc khi so sanh pixel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def eye_region(gray):
    """Lay vung gan vi tri mat de kiem tra thay doi."""
    h, w = gray.shape

    x1 = int(w * EYE_X1)
    x2 = int(w * EYE_X2)
    y1 = int(h * EYE_Y1)
    y2 = int(h * EYE_Y2)

    return gray[y1:y2, x1:x2]


def mean_pixel_diff(a, b):
    """Sai khac pixel trung binh."""
    return float(np.mean(cv2.absdiff(a, b)))


def classify_quality(image):
    """Tra ve danh sach canh bao chat luong."""
    height, width = image.shape[:2]
    warnings = []

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        warnings.append("too_small")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if blur_score < BLUR_THRESHOLD:
        warnings.append("blurry")

    if brightness < DARK_THRESHOLD:
        warnings.append("too_dark")
    elif brightness > BRIGHT_THRESHOLD:
        warnings.append("too_bright")

    return warnings, brightness, blur_score


def copy_image(source, destination_root, category):
    """Sao chep anh vao thu muc theo ly do."""
    destination = destination_root / category / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Tranh ghi de neu trung ten file.
    if destination.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1

        while destination.exists():
            destination = destination_root / category / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(source, destination)
    return destination


# ============================================================
# QUET ANH
# ============================================================

image_paths = sorted(
    p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not image_paths:
    raise RuntimeError(f"Khong tim thay anh trong:\n{INPUT_DIR}")

print("=" * 65)
print(" CHECKSLEEPONE - ADVANCED FRAME FILTER")
print("=" * 65)
print(f"Nguon: {INPUT_DIR}")
print(f"Tong anh: {len(image_paths)}")
print()

# ============================================================
# BIEN THONG KE
# ============================================================

seen_hashes = {}
last_kept_hash = None
last_kept_gray = None
last_kept_eye = None

stats = Counter()
report_rows = []

# ============================================================
# XU LY TUNG FRAME
# ============================================================

for index, source in enumerate(image_paths, start=1):

    status = "kept"
    reason = "OK"
    brightness = ""
    blur_score = ""
    width = ""
    height = ""

    print(f"[{index}/{len(image_paths)}] {source.name}")

    # --------------------------------------------------------
    # DOC ANH
    # --------------------------------------------------------

    try:
        image = cv2.imread(str(source))
    except Exception:
        image = None

    if image is None or image.size == 0:
        copy_image(source, REVIEW_DIR, "corrupt")
        status = "review"
        reason = "corrupt_or_unreadable"

        stats["corrupt"] += 1
        stats["review"] += 1

        report_rows.append(
            [source.name, status, reason, width, height, brightness, blur_score, ""]
        )
        print("  -> REVIEW: anh loi")
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # TRUNG FILE CHINH XAC
    # --------------------------------------------------------

    file_hash = sha256_file(source)

    if file_hash in seen_hashes:
        copy_image(source, REVIEW_DIR, "exact_duplicate")

        status = "review"
        reason = "exact_duplicate_of_" + seen_hashes[file_hash]

        stats["exact_duplicate"] += 1
        stats["review"] += 1

        report_rows.append(
            [source.name, status, reason, width, height, brightness, blur_score, ""]
        )
        print("  -> REVIEW: trung anh chinh xac")
        continue

    # --------------------------------------------------------
    # CHAT LUONG ANH
    # --------------------------------------------------------

    warnings, brightness, blur_score = classify_quality(image)

    if warnings:
        # Khong loai bo khoi du lieu mot cach im lang.
        # Anh can kiem tra duoc copy rieng.
        category = warnings[0]
        copy_image(source, REVIEW_DIR, category)

        status = "review"
        reason = ";".join(warnings)

        for warning in warnings:
            stats[warning] += 1

        stats["review"] += 1
        seen_hashes[file_hash] = source.name

        report_rows.append(
            [
                source.name,
                status,
                reason,
                width,
                height,
                round(brightness, 2),
                round(blur_score, 2),
                "",
            ]
        )
        print(f"  -> REVIEW: {reason}")
        continue

    # --------------------------------------------------------
    # PHAT HIEN FRAME GAN GIONG
    # --------------------------------------------------------

    gray = resize_gray(image)
    current_hash = dhash(image)
    current_eye = eye_region(gray)

    near_duplicate = False
    distance = ""
    whole_diff = ""
    eye_diff = ""

    if ENABLE_NEAR_DUPLICATE_FILTER and last_kept_hash is not None:
        distance = hash_distance(current_hash, last_kept_hash)

        whole_diff = mean_pixel_diff(gray, last_kept_gray)

        eye_diff = mean_pixel_diff(current_eye, last_kept_eye)

        # Chi coi la gan trung khi:
        # 1. dHash gan nhau
        # 2. Toan anh thay doi rat it
        # 3. Vung gan mat cung thay doi rat it
        #
        # Neu vung mat thay doi ro, giu frame.
        near_duplicate = (
            distance <= DHASH_THRESHOLD
            and whole_diff <= MAX_PIXEL_DIFF
            and eye_diff <= MAX_EYE_DIFF_FOR_DUPLICATE
        )

    if near_duplicate:
        stats["near_duplicate"] += 1

        if COPY_NEAR_DUPLICATES_TO_REVIEW:
            copy_image(source, REVIEW_DIR, "near_duplicate")
            status = "review"
            reason = "near_duplicate"
            stats["review"] += 1
        else:
            status = "skipped"
            reason = "near_duplicate"

        report_rows.append(
            [
                source.name,
                status,
                reason,
                width,
                height,
                round(brightness, 2),
                round(blur_score, 2),
                f"dhash={distance}; whole={whole_diff:.2f}; " f"eye={eye_diff:.2f}",
            ]
        )

        print(
            f"  -> {status.upper()}: gan trung "
            f"(dHash={distance}, eye_diff={eye_diff:.2f})"
        )
        continue

    # --------------------------------------------------------
    # GIU FRAME
    # --------------------------------------------------------

    destination = OUTPUT_DIR / source.name

    # Neu dau ra da co file cung ten, tao ten khac.
    if destination.exists():
        destination = OUTPUT_DIR / (f"{source.stem}_{index}{source.suffix}")

    shutil.copy2(source, destination)

    seen_hashes[file_hash] = source.name

    last_kept_hash = current_hash
    last_kept_gray = gray
    last_kept_eye = current_eye

    stats["kept"] += 1

    report_rows.append(
        [
            source.name,
            "kept",
            "OK",
            width,
            height,
            round(brightness, 2),
            round(blur_score, 2),
            f"dhash={distance}; whole={whole_diff}; eye={eye_diff}",
        ]
    )

    print("  -> GIU LAI")

# ============================================================
# XUAT BAO CAO CSV
# ============================================================

with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "filename",
            "status",
            "reason",
            "width",
            "height",
            "brightness",
            "blur_score",
            "comparison",
        ]
    )

    writer.writerows(report_rows)

# ============================================================
# TONG KET
# ============================================================

print()
print("=" * 65)
print(" KET QUA LOC ANH")
print("=" * 65)
print(f"Tong anh ban dau       : {len(image_paths)}")
print(f"Anh giu lai            : {stats['kept']}")
print(f"Trung chinh xac        : {stats['exact_duplicate']}")
print(f"Gan trung              : {stats['near_duplicate']}")
print(f"Anh loi                : {stats['corrupt']}")
print(f"Anh qua nho             : {stats['too_small']}")
print(f"Anh mo                 : {stats['blurry']}")
print(f"Anh qua toi            : {stats['too_dark']}")
print(f"Anh qua sang           : {stats['too_bright']}")
print(f"Tong anh review        : {stats['review']}")
print()
print(f"Anh giu lai: {OUTPUT_DIR}")
print(f"Anh can xem: {REVIEW_DIR}")
print(f"Bao cao CSV: {REPORT_CSV}")
print("=" * 65)
print("HOAN TAT - ANH GOC KHONG BI XOA.")

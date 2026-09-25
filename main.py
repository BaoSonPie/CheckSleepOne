import cv2
from camera.camera import Camera

camera = Camera()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Không đọc được frame")
        break

    cv2.imshow("CheckSleepOne - Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break


camera.release()
cv2.destroyAllWindows()

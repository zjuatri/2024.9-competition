import cv2
import numpy as np
import time 

lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow('USB Camera', hsv_frame)
    mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
    # cv2.imshow('Mask Green', mask_green)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    blurred_image = cv2.GaussianBlur(mask_green, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), 3)
            cv2.imshow('Detected Circles', frame)

    
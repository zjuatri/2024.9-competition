import cv2
import numpy as np
import time

# 定义各种颜色的HSV范围
red_lower_hsv1 = np.array([0, 43, 46])
red_lower_hsv2 = np.array([156, 43, 46])
red_upper_hsv1 = np.array([10, 255, 255])
red_upper_hsv2 = np.array([180, 255, 255])

blue_lower_hsv = np.array([100, 43, 46])
blue_upper_hsv = np.array([124, 255, 255])

yellow_lower_hsv = np.array([26, 43, 46])
yellow_upper_hsv = np.array([34, 255, 255])

black_lower_hsv = np.array([0,0,0])
black_upper_hsv = np.array([180,255,46])
# 定义HSV阈值的初始范围
min_h = 0
max_h = 179
min_s = 0
max_s = 255
min_v = 0
max_v = 255

# 读取图片
img = cv2.imread('./3.jpg')


# 将图片从BGR转换到HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
v = cv2.equalizeHist(v)
hsv = cv2.merge([h,s,v])
#各个掩码
red_mask1 = cv2.inRange(hsv, red_lower_hsv1, red_upper_hsv1)
red_mask2 = cv2.inRange(hsv, red_lower_hsv2, red_upper_hsv2)

red_mask = cv2.bitwise_or(red_mask1, red_mask2)
blue_mask = cv2.inRange(hsv, blue_lower_hsv, blue_upper_hsv)
yellow_mask = cv2.inRange(hsv, yellow_lower_hsv, yellow_upper_hsv)
black_mask = cv2.inRange(hsv,black_lower_hsv, black_upper_hsv)
#回调函数
def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow("Modify")
cv2.createTrackbar('min_h', "Modify", min_h, 179, nothing)
cv2.createTrackbar('max_h', "Modify", max_h, 179, nothing)
cv2.createTrackbar('min_s', "Modify", min_s, 255, nothing)
cv2.createTrackbar('max_s', "Modify", max_s, 255, nothing)
cv2.createTrackbar('min_v', "Modify", min_v, 255, nothing)
cv2.createTrackbar('max_v', "Modify", max_v, 255, nothing)

cv2.namedWindow("final",cv2.WINDOW_NORMAL)
# cv2.namedWindow("HSV",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)

def recognize(mask):
    cv2.imshow("mask",mask)
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # 开操作去除噪点
    image = cv2.morphologyEx(mask,cv2.MORPH_OPEN,element)
    # 闭操作连接连通域
    image = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,element)

    cv2.imshow("final",image)
    cv2.resizeWindow('mask', 800, 540)
    cv2.resizeWindow('final', 800, 540)

while True:
    recognize(red_mask)
    # # 获取当前滑动条的位置
    # min_h = cv2.getTrackbarPos('min_h', 'Modify')
    # max_h = cv2.getTrackbarPos('max_h', 'Modify')
    # min_s = cv2.getTrackbarPos('min_s', 'Modify')
    # max_s = cv2.getTrackbarPos('max_s', 'Modify')
    # min_v = cv2.getTrackbarPos('min_v', 'Modify')
    # max_v = cv2.getTrackbarPos('max_v', 'Modify')

    # # 根据滑动条的位置创建HSV阈值
    # lower_hsv = np.array([min_h, min_s, min_v])
    # upper_hsv = np.array([max_h, max_s, max_v])

    # # 根据阈值创建掩码
    # mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # # 将掩码应用到原始图像
    # result = cv2.bitwise_and(img, img, mask=blue_mask)

    # # 显示所有图像
    # cv2.imshow("RGB", img)
    # #cv2.imshow("HSV", hsv)
    # cv2.imshow("Result", result)
    # cv2.resizeWindow('RGB', 960, 540)
    # #cv2.resizeWindow('HSV', 500, 400)
    # cv2.resizeWindow('Result', 960, 540)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放所有资源
cv2.destroyAllWindows()



# lower_red_1 = np.array([100, 43, 46])
# upper_red_1 = np.array([124, 255, 255])

# cap = cv2.VideoCapture(1)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     cv2.imshow('USB Camera', frame)
#     mask_red_1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
#     cv2.imshow('Mask Red 1', mask_red_1)

#     # time.sleep(1)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     # blurred_image = cv2.GaussianBlur(mask_red_1, (9, 9),0)
#     circles = cv2.HoughCircles(mask_red_1, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
#     # print(circles)
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles[0, :]:
#             center = (circle[0], circle[1])
#             radius = circle[2]
#             cv2.circle(frame, center, radius, (0, 255, 0), 2)
#             cv2.circle(frame, center, 2, (0, 0, 255), 3)
#             cv2.imshow('Detected Circles', frame)

    
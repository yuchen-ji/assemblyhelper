import cv2
import numpy as np

# 读取二值分割图像
image = cv2.imread("VM/labels/crossscrew/cross_2.png")

# 将图像转化为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 通过阈值将灰度图划分为二值图
# 因为是分割后的图像，所以空白区域值为0
threshold_value = 0
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# 寻找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化变量以存储最大包围框信息
max_area = 0
main_direction = 0
main_bbox = None

# 遍历每个轮廓
for contour in contours:
    
    # 计算最小包围框
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    
    # 计算包围框的面积
    area = size[0] * size[1]
    
    # 如果找到更大的包围框，更新主方向和最小包围框
    if area > max_area:
        max_area = area
        main_direction = angle
        main_bbox = rect


# 绘制最小包围框
rotated_box = cv2.boxPoints(main_bbox)
rotated_box = np.int0(rotated_box)
cv2.drawContours(image, [rotated_box], 0, (0, 255, 0), 2)

# @Rect：4.5之后的版本，以bbox的最低点为原点，将x轴顺时针旋转，碰到的第一个边所旋转的角度
# @width：4.5之后的版本，以bbox的最低点为原点，将x轴顺时针旋转，碰到的第一个边为width
print("旋转角度：", main_direction)
print("最小包围框：", main_bbox)

# 显示图像和包围框
cv2.imwrite("bbox_image.png", image)
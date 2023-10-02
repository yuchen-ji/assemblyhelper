import cv2
import numpy as np

def find_nonzero_indices(arr):
    nonzero_indices = []
    for i in range(len(arr)):
        if arr[i] != 0:
            nonzero_indices.append(i)
    return nonzero_indices

img = cv2.imread("assets/mask.png")
height, width, _ = img.shape

if height > width:
    top_line = img[0, :, 0]
    top_points = find_nonzero_indices(top_line)
    top_point = np.mean(top_points)
    # TODO

# 如果是下部被遮挡
else:
    left_line = img[:, 0, 0]
    left_ys = find_nonzero_indices(left_line)
    left_y = np.mean(left_ys)
    left_point = (0, int(left_y))
    
    
    right_line = img[:, width-1, 0]
    right_ys = find_nonzero_indices(right_line)
    right_y = np.mean(right_ys)
    right_point = (width-1, int(right_y))
    
    # 框架的中心点坐标
    center_x = int (width / 2)
    center_y = int ((left_y + right_y) / 2)
    center_point = (center_x, center_y)
    
    
cv2.circle(img, left_point, 20, color=[0, 0, 255], thickness=-1)
cv2.circle(img, right_point, 20, color=[0, 0, 255], thickness=-1)
cv2.circle(img, center_point, 20, color=[0, 255, 0], thickness=-1)
cv2.line(img, left_point, right_point, color=[0, 0, 255], thickness=3)


cv2.imwrite("assets/mask_.png", img)
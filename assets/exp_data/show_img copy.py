import numpy as np
import cv2
import matplotlib.pyplot as plt

img_info = np.load('src/workspaces/part_space6.npy', allow_pickle=True)
color_data = img_info[()]['color']
point_data = img_info[()]['pointcloud']


cv2.imwrite("src/workspaces/part_space6.jpg", color_data)

# 深度图
depth_map = cv2.normalize(point_data[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
depth_map = depth_map.astype(np.uint8)

# 可视化
cv2.imshow('color', color_data)
# cv2.imshow('depth', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
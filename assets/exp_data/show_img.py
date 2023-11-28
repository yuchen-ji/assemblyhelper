import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_coordinates(coordinates):
    x = [point[0] for point in coordinates]
    y = [point[1] for point in coordinates]

    plt.scatter(x, y, marker='o', color='blue')
    plt.title('Scatter Plot of Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


img_info = np.load('src/workspaces/part_space.npy', allow_pickle=True)
color_data = img_info[()]['color']
point_data = img_info[()]['pointcloud']

mask_info = np.load('src/workspaces/stringer_mask.npy', allow_pickle=True)

# 提取mask的区域
stringer1_mask = mask_info[0]
projected_cloud = point_data[:, :, :2]
masked_cloud = projected_cloud[stringer1_mask != 0]


# 1. 根据最大值/最小值的均值，得到中心点(过滤0元素)
data_x = masked_cloud[..., 0]
data_y = masked_cloud[..., 1]
filter_x = data_x[data_x != 0]
filter_y = data_y[data_y != 0]
max_x, min_x = np.max(filter_x), np.min(filter_x)
max_y, min_y = np.max(filter_y), np.min(filter_y)
mean_xy = 1/2*np.array([(min_x+max_x), (min_y+max_y)])


# 2. 获取XY的中位数，得到中心点
fliter_data = masked_cloud[~np.any(masked_cloud == 0, axis=1)]
mean_xy = np.median(fliter_data, axis=0)
print("中间点的下标：", np.where((fliter_data == mean_xy).all(axis=1)))

# 3. 直接取均值
fliter_data = masked_cloud[~np.any(masked_cloud == 0, axis=1)]
mean_xy = np.mean(fliter_data, axis=0)


matching_indices = [0, 0]
min_distance = 1e10
for i in range(projected_cloud.shape[0]):
    for j in range(projected_cloud.shape[1]):
        if stringer1_mask[i][j] == False:
            continue
        distance = np.linalg.norm(projected_cloud[i, j] - mean_xy)
        if distance < min_distance:
            min_distance = distance
            matching_indices = [j, i]

print(matching_indices)
cv2.imwrite("src/workspaces/7.jpg", color_data)
cv2.circle(color_data, matching_indices, radius=5, color=[0,0,255], thickness=-1)

# 深度图
depth_map = cv2.normalize(point_data[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
depth_map = depth_map.astype(np.uint8)

# 可视化
cv2.imshow('color', color_data)
# cv2.imshow('depth', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
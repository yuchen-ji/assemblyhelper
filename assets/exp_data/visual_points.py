import matplotlib.pyplot as plt
import numpy as np



# 加载RGB和pcd数据
img_info = np.load('src/workspaces/part_space.npy', allow_pickle=True)
color_data = img_info[()]['color']
point_data = img_info[()]['pointcloud']
# 加载mask书记
mask_info = np.load('src/workspaces/stringer_mask.npy', allow_pickle=True)
# 提取mask的区域
stringer1_mask = mask_info[3]
projected_cloud = point_data[:, :, :2]
masked_cloud = projected_cloud[stringer1_mask != 0]
# 过滤为0的点8
fliter_data = masked_cloud[~np.any(masked_cloud == 0, axis=1)]

# 获取均值
mean_xy = np.mean(fliter_data, axis=0)

# 调用函数进行可视化
x = [point[0] for point in fliter_data]
y = [point[1] for point in fliter_data]

plt.scatter(x, y, marker='o', color='blue')
plt.scatter(mean_xy[0], mean_xy[1], marker='o', color='red')

plt.title('Scatter Plot of Coordinates')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()



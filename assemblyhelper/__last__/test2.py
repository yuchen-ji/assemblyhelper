import numpy as np
from sklearn.ensemble import IsolationForest

# 生成一些示例数据
# data = np.random.randn(100, 3)  # 生成一个100x3的示例数据
mask_info = np.load('info.npy', allow_pickle=True)[()]
mask = mask_info["masks"][10].astype("uint8")

img_info = np.load('assemblyhelper/streams/capture_2023-10-24-14-57-38.npy', allow_pickle=True)
depth_data = img_info[()]["pointcloud"][:, :, 2]

mask_indices = np.argwhere(mask != 0)
depth_data = depth_data[mask_indices[:, 0], mask_indices[:, 1]].reshape(-1, 1)

# 创建Isolation Forest模型
clf = IsolationForest(contamination=0.05)  # 设置污染率（即预期的离群值比例）

# 训练模型
clf.fit(depth_data)

# 预测离群值
outliers = clf.predict(depth_data)

# outliers的值为1表示正常值，-1表示离群值
print(outliers)

indices = np.argwhere(outliers == 1)
# depth_data = depth_data[indices]
# min_depth = np.min(depth_data)

indices = np.argwhere(depth_data != 0)
depth_data = depth_data[indices]
min_depth = np.min(depth_data)

print(min_depth)

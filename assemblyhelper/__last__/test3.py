import cv2
import numpy as np
from sklearn.ensemble import IsolationForest

mask_info = np.load('info.npy', allow_pickle=True)[()]
# img_info = np.load('assemblyhelper/streams/capture_2023-10-24-14-57-38.npy', allow_pickle=True)[()]
img_info = np.load('assemblyhelper/streams/capture.npy', allow_pickle=True)[()]


grasp_idx = [i for i, v in enumerate(mask_info["classes"]) if v == 'stringer'][0]

mask = mask_info["masks"][grasp_idx].astype("uint8")
color_data = img_info["color"]
depth_data = img_info["pointcloud"][:, :, 2]
projected_cloud = img_info["pointcloud"][:, :, :2]

# 过滤mask==0的index
mask_indices = np.argwhere(mask != 0)
depth_data_ = depth_data[mask_indices[:, 0], mask_indices[:, 1]]
# 过滤深度为0的index
depth_indices = np.argwhere(depth_data_ != 0)
# final_indices = mask_indices[depth_indices].squeeze(1)
# # 再次过滤离群值
depth_data_ = depth_data_[depth_indices]
clf = IsolationForest(contamination=0.5)
clf.fit(depth_data_)
outliers = clf.predict(depth_data_)
outliers_indices = np.argwhere(outliers == 1)
final_indices = mask_indices[depth_indices[outliers_indices].squeeze(1)].squeeze(1)

mean_xy = np.mean(projected_cloud[final_indices[:, 0], final_indices[:, 1]], axis=0)
# mean_xy = np.array()
min_z = np.min(depth_data_)
print(min_z)

selected_indices = [0, 0]
min_distance = 1e10        

for idx in final_indices:
    cv2.circle(color_data, [idx[1], idx[0]], radius=1, color=[255,0,0], thickness=-1)
    distance = np.linalg.norm(projected_cloud[idx[0], idx[1]] - mean_xy)
    if distance < min_distance:
        # if np.linalg.norm(depth_data[idx[0], idx[1]] - min_z) <= 0.005:
            min_distance = distance
            selected_indices = [idx[1], idx[0]] 

cv2.circle(color_data, selected_indices, radius=5, color=[0,255,0], thickness=-1)
cv2.imwrite("1_.png", color_data)

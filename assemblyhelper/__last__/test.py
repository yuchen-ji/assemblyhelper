# a = ["1", '2', '3', '4', '5', '6']

# if '8' in a or '7':
#     print("xxx")

# import cv2
# import numpy as np
# info = np.load("assemblyhelper/streams/capture_2023-10-24-14-57-38.npy", allow_pickle=True)
# color = info[()]["color"]
# cv2.imwrite('assemblyhelper/streams/1.png', color)

# from datetime import datetime
# file = "capture_2023-10-24-14-57-38.npy"
# timestamp = file.split('_')[-1].split('.')[0]
# timestamp = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
# print(timestamp)


# import numpy as np

# # 假设你有一个二值掩码图像，可以表示为一个NumPy数组
# mask = np.array([[0, 1, 0, 0],
#                  [1, 0, 0, 1],
#                  [0, 0, 0, 0],
#                  [0, 1, 1, 0]])

# # 使用numpy.where函数找到不为0的索引
# non_zero_indices = np.where(mask != 0)

# # 打印结果
# print("不为0的索引")
# for index in zip(non_zero_indices[0], non_zero_indices[1]):
#     print(index)


# import numpy as np

# # 假设你有一个名为mask的二值掩码图像
# mask = np.array([[0, 1, 0, 0],
#                  [1, 0, 0, 1],
#                  [0, 0, 0, 0],
#                  [0, 1, 1, 0]])

# # 使用np.argwhere查找非零元素的索引
# non_zero_indices = np.argwhere(mask != 0)

# # non_zero_indices现在包含所有不为0的索引坐标
# print(mask[non_zero_indices[:, 0], non_zero_indices[:, 1]])

# import numpy as np
# import cv2

# img_info = np.load('assemblyhelper/streams/capture_2023-10-24-14-57-38.npy', allow_pickle=True)
# mask_info = np.load('info.npy', allow_pickle=True)[()]

# color_data = img_info[()]['color']
# point_data = img_info[()]['pointcloud']

# # 只计算前两个维度
# projected_cloud = point_data[:, :, :2]

# # 选择mask区域的中心点的indices，和对应的xyz
# grasp_idx = [i for i, v in enumerate(mask_info["classes"]) if v == 'stringer'][1]
# print(grasp_idx)
# bin_image = mask_info["masks"][grasp_idx].astype("uint8") * 255

# mask_indices = np.argwhere(bin_image != 0)
# mean_xy = np.mean(projected_cloud[mask_indices[:, 0], mask_indices[:, 1]], axis=0)
# # mean_xy = np.array([mean_xy[0], mean_xy[1], 0])
# # min_z = np.min(point_data[:, :, 2])
# min_z = 0.444

# selected_indices = [0, 0]
# min_distance = 1e10        
# # for i in range(projected_cloud.shape[0]):
# #     for j in range(projected_cloud.shape[1]):
# #         distance = np.linalg.norm(projected_cloud[i, j] - mean_xy)
# #         if distance < min_distance:
# #             min_distance = distance
# #             selected_indices = [j, i]
# for idx in mask_indices:
#     distance = np.linalg.norm(projected_cloud[idx[0], idx[1]] - mean_xy)
#     if distance < min_distance:
#         if np.linalg.norm(point_data[idx[0], idx[1]][2] - min_z) <= 0.0005:
#             min_distance = distance
#             selected_indices = [idx[1], idx[0]] 

# # print(projected_cloud[32, 507])
# print(point_data[42, 221])
# print(point_data[idx[0], idx[1]])       
# # print(min_distance)
# cv2.circle(color_data, selected_indices, radius=5, color=[0,0,255], thickness=-1)
# cv2.circle(color_data, [221, 42], radius=5, color=[0,0,255], thickness=-1)
# cv2.imwrite("1_.png", color_data)


import cv2
import numpy as np
img_info = np.load('assemblyhelper/streams/capture.npy', allow_pickle=True)[()]
color_img = img_info["color"]
cv2.imwrite("1.png", color_img)


import numpy as np

# 创建一个示例的二维数组
array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

# 提取右下角4个元素
bottom_right = array[-3:, -3:]

print("右下角4个元素:")
print(bottom_right)


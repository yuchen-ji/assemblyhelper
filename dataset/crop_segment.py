import os
import cv2
import numpy as np

# 输入文件夹和输出文件夹的路径
input_folder = "/workspaces/assemblyhelper/testset_final/testset_final"
output_folder = "/workspaces/assemblyhelper/testset_final/output_images"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构建完整的文件路径
    input_path = os.path.join(input_folder, filename)

    # 读取图片
    image = cv2.imread(input_path)

    # 进行相同的操作
    mask = np.zeros_like(image)
    mask[image != 0] = 1

    mask_mean = np.mean(mask, axis=2)
    y, x = np.where(mask_mean)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    crop_img = image[ymin:ymax, xmin:xmax, :]

    # 构建输出文件路径
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")

    # 保存处理后的图片为JPEG格式
    cv2.imwrite(output_path, crop_img)


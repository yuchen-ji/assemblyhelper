import cv2
import numpy as np

# # todo 读取并转换图片格式
# opencv = cv2.imread('dataset/trainset_ori/phillips2/2222.jpg')
# hsv = cv2.cvtColor(opencv, cv2.COLOR_BGR2HSV)

# # todo 指定绿色范围,60表示绿色，我取的范围是-+10
# minGreen = np.array([45, 85, 80])
# maxGreen = np.array([75, 255, 255])

# # todo 确定绿色范围
# mask = cv2.inRange(hsv, minGreen, maxGreen)

# # todo 确定非绿色范围
# mask_not = cv2.bitwise_not(mask)

# # todo 通过掩码控制的按位与运算锁定绿色区域
# green = cv2.bitwise_and(opencv, opencv, mask=mask)

# # todo 通过掩码控制的按位与运算锁定非绿色区域
# green_not = cv2.bitwise_and(opencv, opencv, mask=mask_not)

# # todo 拆分为3通道
# b, g, r = cv2.split(green_not)

# # todo 合成四通道
# bgra = cv2.merge([b, g, r, mask_not])

# # todo 保存带有透明通道的png图片,有了这种素材之后，就可以给这张图片替换任意背景了
# # cv2.imwrite('ouput.png', bgra)

# y, x = np.where(mask_not)
# xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
# cropped_img = bgra[ymin:ymax, xmin:xmax]

# cv2.imwrite('ouput.png', cropped_img)

import numpy as np

np.random.seed(0)
# 随机生成一个512x512的矩阵
random_matrix = np.random.rand(512, 512)
# 对随机矩阵进行QR分解，得到正交矩阵Q
Q, _ = np.linalg.qr(random_matrix)
# 选择前256列作为基变换矩阵
B = Q[:, :256]
# 创建一个512维的随机向量
v_512 = np.random.rand(512)
# 进行基变换和降维操作
v_256 = np.dot(v_512, B)

# 打印结果
print("原始向量 (512维):", v_512[0])
print("基变换后向量 (256维):", v_256.shape)

import numpy as np
import cv2
from PIL import Image

3024, 4032
img1 = cv2.imread("dataset/testset/stringer/IMG_4761.JPG")
img1 = cv2.resize(img1, (int(img1.shape[1]*2/3), int(img1.shape[0]*2/3)))
cv2.imwrite("output.jpg", img1)

# 4032, 3024
img2 = cv2.imread("dataset/testset/warehouse/IMG_4734.JPG")

# 3024, 4032
img1_np = np.array(img1)
# 4032, 3024
img2_np = np.array(img2)


# img3 = Image.open("dataset/testset/stringer/IMG_4761.JPG")
# img4 = Image.open("dataset/testset/warehouse/IMG_4734.JPG")
# # 3024, 4032
# img3_np = np.array(img3)
# # 4032, 3024
# img4_np = np.array(img4)


# import os
# lis = os.listdir("dataset/frames/battery")
# print(len(lis))

# import os
# import cv2
# import numpy as np
# names = os.listdir("dataset/tmp")
# for name in names:
#     image = cv2.imread(os.path.join("dataset/tmp", name))
#     mask = np.zeros_like(image)
#     mask[image != 0] = 1
#     mask_mean = np.mean(mask, axis=2)
#     y, x = np.where(mask_mean)
#     xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
#     crop_img = image[ymin:ymax, xmin:xmax, :]
#     cv2.imwrite(os.path.join("dataset/tmp", name), crop_img)


# import os
# ori_names = os.listdir("dataset/trainset_ori/hex")
# pro_names = os.listdir("dataset/test")
# for idx, p in enumerate(pro_names):
#     p, _ = os.path.splitext(p)
#     pro_names[idx] = p

# last = []
# for o in ori_names:
#     o, _ = os.path.splitext(o)
#     if o in set(pro_names):
#         continue
#     last.append(int(o))

# last.sort()
# print(last)




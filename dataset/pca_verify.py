import os
import torch
import numpy as np
from sklearn.decomposition import PCA


label_dir = "dataset/labels"
fnames = "battery1"
feature = torch.load(os.path.join(label_dir, fnames, "feature.pth")).numpy()

# 送入PCA的数据并不需要事先做标准化
# mean = np.mean(feature, axis=0)
# std = np.std(feature, axis=0)
# feature = (feature - mean[np.newaxis, :]) / std[np.newaxis, :]
# print(np.linalg.norm(feature[0]))

pca = PCA(n_components=256)
feature = feature.reshape(-1, 512)
pca.fit(feature)

print(len(pca.components_))

# 由api将原始数据X映射到新的维度
transformed_feature_api = pca.transform(feature)

# 手动对原始数据X做线性映射，结果与API是一样的
min_var_direction = pca.components_[:, :]    
norm_feature = feature - pca.mean_
transformed_feature_hand = np.dot(norm_feature, min_var_direction.T)
# transformed_feature_hand /= np.sqrt(pca.explained_variance_)

print(transformed_feature_hand == transformed_feature_api)

# PCA得到的特征是没有经过归一化的
print(np.linalg.norm(transformed_feature_hand[0]))
print(np.linalg.norm(feature[0]))


# 使用物体A得到的trans的相似度比较（
# （原始的feature, 物体1的transformed-feature）
# ==============0================
# 0.912
# 0.9964657713479645
# ==============1================
# 0.959
# 0.9642576886165001
# ==============2================
# 0.8877
# 0.9300047753828703
# ==============3================
# 0.928
# 0.9442017781167156
# ==============4================
# 0.937
# 0.933132438924544
# ==============5================
# 0.9355
# 0.9429549790630537


# （原始的feature, 物体分别的transformed-feature）
# ==============0================
# 0.912
# 0.9964657713479645
# ==============1================
# 0.959
# 0.9986384294293302
# ==============2================
# 0.8877
# 0.9914127053871585
# ==============3================
# 0.928
# 0.9986600131746587
# ==============4================
# 0.937
# 0.997838352829172
# ==============5================
# 0.9355
# 0.9980567801029474
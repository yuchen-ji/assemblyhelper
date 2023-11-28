import os
import torch
import numpy as np


# 将每个sample提取的PCA特征分别与对应的PCA的类别作余弦相似度
label_dir = "dataset/dataset_final/trainset"
test_dir = "dataset/dataset_final/testset"
lnames = os.listdir(label_dir)


label_names = []
tranforms_pca = []
features_pca = []
features_ori = []
for ln in lnames:
    label = np.load(os.path.join(label_dir, ln, "label_256_384.npy"), allow_pickle=True)
    feature_ori = label[()]['feature_ori']
    feature_pca = label[()]['feature_pca']
    transform = label[()]['trans_pca']
    # 对平均特征值归一化
    feature_pca /= np.linalg.norm(feature_pca, axis=-1)
    feature_ori /= np.linalg.norm(feature_ori, axis=-1)
    features_ori.append(feature_ori)
    features_pca.append(feature_pca)
    tranforms_pca.append(transform)
    label_names.append(ln)


# 读取测试集图像，作余弦相似度 (使用PCA后)
tcates = os.listdir(test_dir)
accuracies = []
errors = []
for idx, tc in enumerate(tcates):
    similarities = []
    img_features = torch.load(os.path.join(test_dir, tc, "feature.pth"))
    label_idx = label_names.index(tc)
    print(f"=============={tc}===============")
    for t, f in zip(tranforms_pca, features_pca):
        transformed_img_features = np.dot(img_features, t.T)
        transformed_img_features /= np.linalg.norm(transformed_img_features, axis=-1)[:, np.newaxis]
        similarity = transformed_img_features @ f[:, np.newaxis]
        similarities.append(similarity)
        # print(1-np.mean(similarity, axis=0))
        
    # 在得到到N组（类别）余弦相似度值中，选择最大的
    similarities = np.concatenate(similarities, axis=1)
    pred = np.argmax(similarities, axis=1)
    label = np.array([label_idx for _ in range(len(pred))])
    
    # 计算准确率
    result = np.equal(pred, label)
    accuracy = np.sum(result) / len(pred)
    accuracies.append(accuracy)
    errors.append(len(result) - np.sum(result))
    
    # 错误的样本
    indices = np.where(result == False)
    samples = os.listdir(os.path.join(test_dir, tc))
    samples = np.array([s for s in samples if s.endswith('.jpg') or s.endswith('.png')])
    print(samples[indices])
    
print(accuracies)
print(errors)



# 读取测试集图像，作余弦相似度 (原有的512维特征)
tcates = os.listdir(test_dir)
accuracies = []
errors = []
for idx, tc in enumerate(tcates):
    similarities = []
    img_features = torch.load(os.path.join(test_dir, tc, "feature.pth")).numpy()
    label_idx = label_names.index(tc)
    print(f"=============={tc}===============")
    for f in features_ori:       
        similarity = img_features @ f[:, np.newaxis]
        similarities.append(similarity)
        # print(1-np.mean(similarity, axis=0))
        
    # 在得到到N组（类别）余弦相似度值中，选择最大的
    similarities = np.concatenate(similarities, axis=1)
    pred = np.argmax(similarities, axis=1)
    label = np.array([label_idx for _ in range(len(pred))])
    
    # 计算准确率
    result = np.equal(pred, label)
    accuracy = np.sum(result) / len(pred)
    accuracies.append(accuracy)
    errors.append(len(result) - np.sum(result))
    
    
    # 错误的样本
    indices = np.where(result == False)
    samples = os.listdir(os.path.join(test_dir, tc))
    samples = np.array([s for s in samples if s.endswith('.jpg') or s.endswith('.png')])
    print(samples[indices])
    
print(accuracies)
print(errors)

# ['hex', 'warehouse', 'stringer', 'hand', 'phillips', 'slotted', 'signalinterfaceboard', 'battery', 'framework']

# ORIGIN
# [0, 0, 15, 0, 1, 0, 5, 1, 1]

# PCA-512
# [0, 0, 15, 0, 1, 0, 5, 1, 1] = 23

# PCA-256
# [0, 0, 5, 0, 0, 0, 0, 1, 0] = 6

# PCA-128
# [0, 1, 6, 0, 0, 0, 1, 2, 1] = 11


# ==============hex===============
# ['2031.png' '2030.png' '2029.png' '2040.png']
# ==============warehouse===============
# ['IMG_4757.jpg']
# ==============stringer===============
# ['IMG_4787.jpg' 'IMG_4776.jpg' 'IMG_4771.jpg' 'IMG_4793.jpg'
#  'IMG_4777.jpg' 'IMG_4788.jpg' 'IMG_4781.jpg']
# ==============hand===============
# []
# ==============phillips===============
# []
# ==============slotted===============
# []
# ==============signalinterfaceboard===============
# ['IMG_4807.jpg']
# ==============battery===============
# ['IMG_4711.jpg' 'IMG_4701.jpg']
# ==============framework===============
# []



# ==============hex===============
# []
# ==============warehouse===============
# []
# ==============stringer===============
# ['IMG_4787.jpg' 'IMG_4772.jpg' 'IMG_4771.jpg' 'IMG_4784.jpg'
#  'IMG_4791.jpg' 'IMG_4774.jpg' 'IMG_4773.jpg' 'IMG_4794.jpg'
#  'IMG_4793.jpg' 'IMG_4766.jpg' 'IMG_4777.jpg' 'IMG_4788.jpg'
#  'IMG_4781.jpg']
# ==============hand===============
# []
# ==============phillips===============
# ['2008.png']
# ==============slotted===============
# []
# ==============signalinterfaceboard===============
# ['IMG_4823.jpg']
# ==============battery===============
# ['IMG_4701.jpg']
# ==============framework===============
# ['IMG_4858.jpg']


# 没有办法通过PCA得到使类别方向相差最大的方向，因为只有很少的类别
# 使用均值feature得到的余弦相似度
# ==============warehouse===============
# [0.00107726]
# [0.16643679]
# [0.21650353]
# [0.08828928]
# [0.11100201]
# ==============stringer===============
# [0.22236016]
# [0.00176257]
# [0.1633566]
# [0.12878141]
# [0.16477719]
# ==============signalinterfaceboard===============
# [0.1687194]
# [0.09923306]
# [0.00420596]
# [0.12813362]
# [0.17808236]
# ==============battery===============
# [0.1689192]
# [0.12544321]
# [0.19839591]
# [0.00134098]
# [0.05872221]
# ==============framework===============
# [0.15808313]
# [0.14062057]
# [0.23246011]
# [0.08041019]
# [0.00092231]
# [1.0, 1.0, 1.0, 1.0, 1.0]


# ==============warehouse===============
# [0.03174]
# [0.2129]
# [0.1821]
# [0.1914]
# [0.1772]
# ==============stringer===============
# [0.2231]
# [0.04492]
# [0.1367]
# [0.1738]
# [0.1831]
# ==============signalinterfaceboard===============
# [0.2041]
# [0.148]
# [0.05762]
# [0.2192]
# [0.2026]
# ==============battery===============
# [0.1953]
# [0.1665]
# [0.2017]
# [0.03662]
# [0.1162]
# ==============framework===============
# [0.1758]
# [0.1704]
# [0.1792]
# [0.10986]
# [0.03027]
# [1.0, 0.993103448275862, 1.0, 1.0, 1.0]
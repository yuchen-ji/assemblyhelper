import os
import torch
import numpy as np

# each sample's feature should multiply by PCA's transformation of each categories,
# then calculate the cosine similarity between the sample features after PCA and the features of each category.
label_dir = "dataset/trainset"
test_dir = "dataset/testset"
lnames = os.listdir(label_dir)

# get each category's features label (origin, PCA-128, PCA-256, ...) and the PCA transformation 
label_names = []
tranforms_pca = []
features_pca = []
features_ori = []
for ln in lnames:
    label = np.load(os.path.join(label_dir, ln, "label_256_384.npy"), allow_pickle=True)
    feature_ori = label[()]['feature_ori']
    feature_pca = label[()]['feature_pca']
    transform = label[()]['trans_pca']
    # normalize the feature
    feature_pca /= np.linalg.norm(feature_pca, axis=-1)
    feature_ori /= np.linalg.norm(feature_ori, axis=-1)
    features_ori.append(feature_ori)
    features_pca.append(feature_pca)
    tranforms_pca.append(transform)
    label_names.append(ln)


# read test images, make cosin similarity after PCA
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
        
    # get the max similarity as categories
    similarities = np.concatenate(similarities, axis=1)
    pred = np.argmax(similarities, axis=1)
    label = np.array([label_idx for _ in range(len(pred))])
    
    # calculate the accuracy of each category
    result = np.equal(pred, label)
    accuracy = np.sum(result) / len(pred)
    accuracies.append(accuracy)
    errors.append(len(result) - np.sum(result))
    
    # get error samples
    indices = np.where(result == False)
    samples = os.listdir(os.path.join(test_dir, tc))
    samples = np.array([s for s in samples if s.endswith('.jpg') or s.endswith('.png')])
    print(samples[indices])
    
print(accuracies)
print(errors)



# get test images, make cosin similarity with origin 512 feature.
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
        
    # get the max probability as the samples' category
    similarities = np.concatenate(similarities, axis=1)
    pred = np.argmax(similarities, axis=1)
    label = np.array([label_idx for _ in range(len(pred))])
    
    # calculate the accuracy
    result = np.equal(pred, label)
    accuracy = np.sum(result) / len(pred)
    accuracies.append(accuracy)
    errors.append(len(result) - np.sum(result))
    
    # print error samples
    indices = np.where(result == False)
    samples = os.listdir(os.path.join(test_dir, tc))
    samples = np.array([s for s in samples if s.endswith('.jpg') or s.endswith('.png')])
    print(samples[indices])
    
print(accuracies)
print(errors)
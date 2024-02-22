import os
import mmcv
import numpy as np
from PIL import Image
import cv2
import copy
from segment_anything import build_sam, SamAutomaticMaskGenerator
from rembg import remove, new_session
import torch
import clip
from sklearn.decomposition import PCA


def extract_frames(video_path, frame_dir, fnum=300):
    """
    extract frames from a video
    """
    video_reader = mmcv.VideoReader(video_path)
    vname, _ = os.path.splitext(os.path.basename(video_path))
    os.makedirs(os.path.join(frame_dir, vname), exist_ok=True)
    
    interval = video_reader.frame_cnt // fnum
    for i, frame in enumerate(video_reader):
        if i % interval != 0:
            continue
        fname = f"{os.path.join(frame_dir, vname)}/{i:04d}.jpg"
        mmcv.imwrite(frame, fname)


def background_remove2(image_path, label_dir):
    opencv = cv2.imread(image_path)
    hsv = cv2.cvtColor(opencv, cv2.COLOR_BGR2HSV)
    # This setting is for framework
    # minGreen = np.array([30, 95, 85])
    # maxGreen = np.array([80, 255, 255])
    
    # This setting is for screwdriver
    minGreen = np.array([45, 85, 80])
    maxGreen = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, minGreen, maxGreen)
    mask_not = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(opencv, opencv, mask=mask)
    green_not = cv2.bitwise_and(opencv, opencv, mask=mask_not)
    b, g, r = cv2.split(green_not)
    bgra = cv2.merge([b, g, r, mask_not])
    
    y, x = np.where(mask_not)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    cropped_img = bgra[ymin:ymax, xmin:xmax]
    
    cname = os.path.basename(os.path.dirname(image_path))
    os.makedirs(os.path.join(label_dir, cname), exist_ok=True)
    cv2.imwrite(os.path.join(label_dir, cname, os.path.basename(image_path)), cropped_img)


def background_remove(session, image_path, label_dir):
    """
    remvoe background
    """
    
    input =Image.open(image_path)
    output=remove(input, session=session)
    outnp = np.asarray(output)
    
    cname = os.path.basename(os.path.dirname(image_path))
    os.makedirs(os.path.join(label_dir, cname), exist_ok=True)
    # output.save(os.path.join(label_dir, cname, os.path.basename(image_path)))  
    
    # get the minimum bbox of object
    mask = np.zeros_like(outnp)
    mask[outnp != 0] = 1
    mask = mask[:, :, 3]
    y, x = np.where(mask)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    crop_img = outnp[ymin:ymax, xmin:xmax, :]
    
    output = Image.fromarray(crop_img).convert("RGB")
    cname = os.path.basename(os.path.dirname(image_path))
    os.makedirs(os.path.join(label_dir, cname), exist_ok=True)
    output.save(os.path.join(label_dir, cname, os.path.basename(image_path)))    


def pad_to_square(input_image, fill_color=(0, 0, 0)):
    width, height = input_image.size
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), fill_color)
    
    left = (new_size - width) // 2
    top = (new_size - height) // 2    
    
    new_image.paste(input_image, (left, top))
    return new_image
    

@torch.no_grad()
def extract_feature(image_dir, preprocess, model):
    """
    extract features using CLIP
    """
    images = []
    inames = os.listdir(image_dir)
    for im in inames:
        _, ext_name = os.path.splitext(im)
        if ext_name != ".jpg" and ext_name != ".png":
            continue
        images.append(pad_to_square(Image.open(os.path.join(image_dir, im))))
        imm = pad_to_square(Image.open(os.path.join(image_dir, im)))
        # imm.save("output.jpg")
    
    processed_imgs = torch.stack([preprocess(im).to("cuda") for im in images])
    img_feature = model.encode_image(processed_imgs)
    img_feature /= img_feature.norm(dim=-1, keepdim=True)
    
    img_feature = img_feature.detach().cpu()
    torch.save(img_feature, os.path.join(image_dir, "feature.pth")) 


def calculate_similarity(feature):
    # calculate the similarity between each two features in the dataset
    sims = []
    for i in range(len(feature)):
        view_1 = feature[i]
        for j in range(i+1, len(feature)):
            view_2 = feature[j]
            sim = view_1 @ view_2.T
            sims.append(sim)
            
    sims = np.stack(sims)
    sims_mean = np.mean(sims)
    print(sims_mean) 


class minPCA():
    def __init__(self, n_components=256):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def get_pca_transform(self, feature, dimension):
        feature = feature.reshape(-1, 512)
        self.pca.fit(feature)
        min_var_direction = self.pca.components_[dimension[0]:dimension[1], :]
        return min_var_direction

    def transform_feature(self, feature, transform):
        # norm_feature = feature - self.pca.mean_
        norm_feature = feature
        transformed_feature = np.dot(norm_feature, transform.T)
        # normalized the feature after PCA transformation
        transformed_feature /= np.linalg.norm(transformed_feature, axis=-1)[:, np.newaxis]
        return transformed_feature

    
if __name__ == '__main__':
    
    # video_path = "dataset/videos/stringer.mp4"
    video_dir = "dataset/trainset_vids"
    frame_dir = "dataset/trainset_ori"
    label_dir = "dataset/dataset_final/trainset"
    test_dir = "dataset/dataset_final/testset"
    tmp_dir = "dataset/testset_meta"
    
    # write videos as frames
    # vnames = os.listdir(video_dir)
    # vnames = ['stringer.mp4', 'signalinterfaceboard.mp4', 'warehouse.mp4', 'framework.mp4']
    # vnames = ["battery3.mp4"] # 225
    # vnames = ["battery4.mp4"] # 225
    # vnames = ["battery5.mp4"]   # 150
    # vnames = ["slotted.mp4", "phillips.mp4"]
    # # vnames= ["slotted2.mp4"]
    # # vnames = ["phillips2.mp4"]
    # vnames = ["slotted.mp4", "phillips.mp4"]
    # vnames = ["hand.mp4"]
    # for vn in vnames:
    #     vpath = os.path.join(video_dir, vn)
    #     extract_frames(vpath, frame_dir, fnum=600)
    
    # read all images(frames)
    # session = new_session("u2net")
    # cnames = os.listdir(frame_dir)
    # cnames = ["signalinterfaceboard", "stringer"] # u2net
    # cnames = ["battery", "framework", "warehouse"] # green
    # cnames = ["hex", "slotted", "phillips"]
    # cnames = ["slotted", "phillips"]
    # cnames = ["hand"]
    # for cn in cnames:
    #     frnames = os.listdir(os.path.join(frame_dir, cn))
    #     for fn in frnames:
            # background_remove(session, os.path.join(frame_dir, cn, fn), label_dir)
            # background_remove2(os.path.join(frame_dir, cn, fn), label_dir)
    
    
    # extract features using PCA
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # # lnames = ["warehouse"]
    # # lnames = ["slotted"]
    # lnames = os.listdir(test_dir)
    # for ln in lnames:
    #     extract_feature(os.path.join(test_dir, ln), preprocess, model)
    
    # extract view independent feature using PCA
    trans = []
    fnames = os.listdir(label_dir)
    for idx, fn in enumerate(fnames):
        mpca = minPCA(n_components=512)
        feature = torch.load(os.path.join(label_dir, fn, "feature.pth")).numpy()
        min_var_direction = mpca.get_pca_transform(feature, dimension=[256, 256+128])
        trans.append(min_var_direction)
        transformed_feature = mpca.transform_feature(feature, min_var_direction)
        
        trans_mean_feature  = np.mean(transformed_feature, axis=0)
        trans_mean_feature /= np.linalg.norm(trans_mean_feature)
        
        feature_mean = np.mean(feature, axis=0)
        feature_mean /= np.linalg.norm(feature_mean)

        label = {}
        label['feature_ori'] = feature_mean
        label['feature_pca'] = trans_mean_feature
        label['trans_pca'] = min_var_direction
        np.save(os.path.join(label_dir, fn, "label_256_384.npy"), label)
        

        # print(f"=============={idx}================")
        # calculate_similarity(feature)
        # calculate_similarity(transformed_feature)
    
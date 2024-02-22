import os
import mmcv
import numpy as np
from PIL import Image
import cv2
import copy
import torch
from segment_anything import build_sam, SamAutomaticMaskGenerator


def extract_frames(video_path, frame_dir, fnum=256):
    """
    extract videos as frames
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


def convert_box_xywh_to_xyxy(box):
    """
    reformat bounding box
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [int(x1), int(y1), int(x2), int(y2)]


def segment_image(image, segmentation_mask):
    """
    crop segmented image from origin images
    """
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]  
    black_image = np.zeros_like(image_array)
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    black_image[segmentation_mask] = segmented_image_array[segmentation_mask]
    
    return black_image


def background_det(mask):
    """
    check whether mask is background?
    """
    top_left = np.sum(mask[0:50, 0:50])
    top_right = np.sum(mask[0:50, -50:])
    bottom_left = np.sum(mask[-50:, :50])
    bottom_right = np.sum(mask[-50:, -50:])
    
    if top_left >= 250:
        return True
    if top_right >= 250:
        return True
    if bottom_left >= 250:
        return True
    if bottom_right >= 250:
        return True
    
    return False  


@torch.no_grad()
def data_genrate(mask_generator, image_path, output_dir):
    """
    generates masks based on original image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 4032, 3024
    image = cv2.resize(image, (int(image.shape[1]*2/3), int(image.shape[0]*2/3)))
    masks = mask_generator.generate(image)
    
    # Cut out all masks and each mask's bboxes
    for idx, mask in enumerate(masks):
        x1, y1, x2, y2 = convert_box_xywh_to_xyxy(mask["bbox"])
        cropped_img = segment_image(image, mask["segmentation"])[y1:y2, x1:x2]     
        # if the cropped image is background, filter it.
        # if background_det(mask["segmentation"]):
        #     continue
        cname = os.path.basename(os.path.dirname(image_path))
        iname, _ = os.path.splitext(os.path.basename(image_path))
        os.makedirs(os.path.join(output_dir, cname), exist_ok=True)
        os.makedirs(os.path.join(output_dir, cname, iname), exist_ok=True)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, cname, iname, f"{idx}.jpg"), cropped_img)
        
    del masks
    del mask_generator
    
    
if __name__ == '__main__':
    
    # video_path = "dataset/videos/stringer.mp4"
    video_dir = "dataset/trainset_vids"
    input_dir = "dataset/trainset_ori"
    output_dir = "dataset/tmp"
    
    # write all videos as frames
    # vnames = os.listdir(video_dir)
    # for vn in vnames:
    #     vpath = os.path.join(video_dir, vn)
    #     extract_frames(vpath, frame_dir)
    
    # load sam segmentator
    device = "cuda:0"  
    sam = build_sam(checkpoint="thirdparty/sam_vit_h_4b8939.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=0.95, crop_n_layers=0)
    
    # run sam on each category
    # cnames = os.listdir(test_dir)
    # cnames = ['stringer', 'signalinterfaceboard', 'battery', 'framework']
    cnames = ["hex", "slotted", "phillips"]
    # cnames = ["slotted2"]
    for cn in cnames:      
        frnames = os.listdir(os.path.join(input_dir, cn))
        for fn in frnames:
            data_genrate(mask_generator, os.path.join(input_dir, cn, fn), output_dir)
            torch.cuda.empty_cache()    

# nohup python dataset/dataset_generate2.py > output.log 2>&1 &
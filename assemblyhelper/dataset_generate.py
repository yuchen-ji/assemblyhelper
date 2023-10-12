import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip
import torch
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple
import copy
from pathlib import Path


def process_img(image_path):
    """
    resize and rewrite the image
    """
    im = cv2.imread(image_path)
    w_h = 512
    height, width, _ = im.shape
    ratio = max(width, height) / w_h
    new_width = int(width / ratio)
    new_height = int(height / ratio)
    im = cv2.resize(im, (new_width, new_height))

    # print(f"origin: {width},{height}")
    # print(f"resize: {new_width},{new_height}")
    cv2.imwrite(image_path, im)
    
    
def convert_box_xywh_to_xyxy(box):
    """
    调整bbox的格式
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
    """
    根据分割图将原图的区域crop出来
    """
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    
    return black_image


def get_indices_of_values_above_threshold(values, threshold):
    """
    使用阈值过滤置信度低的分类结果
    """
    indices = [-1] * values.shape[0]
    for i, n in enumerate(values):
        indice = torch.argmax(n)
        if n[indice] > threshold:
            indices[i] = indice      
    return indices


def overlay_image(image_path, indices, masks) -> None:
    """
    将mask不同的类别绘制成不同的颜色
    """
    original_image = Image.open(image_path)
    overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    overlay_color_map = [
        (255, 0, 0, 200),  # Red
        (0, 255, 0, 200),  # Green
        (0, 0, 255, 200),  # Blue
        (255, 255, 0, 200),  # Yellow
        (0, 255, 255, 200),  # Cyan
    ]
    
    draw = ImageDraw.Draw(overlay_image)
    for seg_idx, value in enumerate(indices):
        if value == -1:
            continue
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color_map[value])

    result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)


def data_genrate(mask_generator, image_path, class_name, image_idx):
    """
    检测整个场景，输入整张图像，输出该场景中符合数据集类别的信息，包括（中心点，边界框，类别，分割图）
    """
    
    # Generate masks
    process_img(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    # Cut out all masks and each mask's bboxes
    image = Image.open(image_path)
    cropped_imgs, cropped_bboxes = [], []
    for idx, mask in enumerate(masks):
        cropped_imgs.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))
        cropped_bboxes.append(mask["bbox"])        
        # Save masked single objects
        
        path = Path(f"/workspaces/assemblyhelper/cropped/{class_name}/{image_idx}")
        path.mkdir(parents=True, exist_ok=True)
        cropped_image = copy.deepcopy(cropped_imgs[idx]).convert("RGBA")
        cropped_image.save(os.path.join(path, f"{idx}.png"), "PNG")

    
if __name__ == "__main__":
    
    # Load SAM
    device = "cuda"
    sam = build_sam(checkpoint="VM/sam_vit_h_4b8939.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.98, crop_n_layers=0)
        
    # ori_foler = "/workspaces/assemblyhelper/origin"
    # categories = os.listdir(ori_foler)
    # for cate in categories:
    #     cate_folder = os.path.join(ori_foler, cate, 0)
    #     names = os.listdir(cate_folder)
    #     for idx, img_name in enumerate(names):
    #         img_path = os.path.join(cate_folder, img_name)
    #         detect(mask_generator, img_path, cate, idx)         

    ori_foler = "/workspaces/assemblyhelper/origin/stringer"
    names = os.listdir(ori_foler)
    for idx, img_name in enumerate(names):
        img_path = os.path.join(ori_foler, img_name)
        data_genrate(mask_generator, img_path, "stringer", idx)
    
        
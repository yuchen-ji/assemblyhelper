import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip
import torch
import numpy as np
import os


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
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
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
    return [i for i, v in enumerate(values) if v > threshold]


def detect(image_path, obj_name):    
    # Load SAM
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    
    # Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    process_img(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    # Cut out all masks and each mask's bboxes
    image = Image.open(image_path)
    cropped_imgs, cropped_bboxes = [], []
    for mask in masks:
        cropped_imgs.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))
        cropped_bboxes.append(mask["bbox"])
    
    # Use clip to identify
    @torch.no_grad()
    def retriev(elements, search_label: str) -> int:
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        preprocessed_labels = [preprocess(label).to(device) for label in search_label]
        stacked_images = torch.stack(preprocessed_images)
        stacked_labels = torch.stack(preprocessed_labels)
        image_features = model.encode_image(stacked_images)
        label_features = model.encode_image(stacked_labels)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        label_features /= label_features.norm(dim=-1, keepdim=True)
        label_features = torch.mean(label_features, dim=0, keepdim=True)        
        # print(f"image_shape:{image_features.shape}")
        # print(f"text_shape:{label_features.shape}")
        probs = 100. * image_features @ label_features.T
        return probs[:, 0].softmax(dim=0)
    
    label_dir = ""
    labels = []
    if obj_name:
        cls_list = [obj_name]
    else:
        cls_list = os.listdir(label_dir)
            
    for cls in cls_list:
        label = []
        path = os.path.join(label_dir, cls)
        imgs = os.listdir(path)
        for img in imgs:
            label.append(Image.open(os.path.join(path, img)))
        labels.append(label)
    
    obs = {}
    for cls, label in zip(cls_list, labels):
        scores = retriev(cropped_imgs, label)
        indices = get_indices_of_values_above_threshold(scores, 0.05)
        
        
    # return obj's bboxes
    
    
    
    
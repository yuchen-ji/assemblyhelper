import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip
import torch
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple


class OpenDetector:
    def __init__(
        self,
        label_dir,
        checkpoint="/workspaces/assemblyhelper/thirdparty/sam_vit_h_4b8939.pth",
    ):
        """
        初始化模型
        """
        # Load SAM
        device = "cuda"
        sam = build_sam(checkpoint=checkpoint).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam, points_per_side=32, pred_iou_thresh=0.98, crop_n_layers=0
        )

        # Load CLIP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_dir = label_dir

    def process_img(self, image_path) -> None:
        """
        resize and rewrite the image
        """
        im = cv2.imread(image_path)
        w_h = 640
        height, width, _ = im.shape
        ratio = max(width, height) / w_h
        new_width = int(width / ratio)
        new_height = int(height / ratio)
        im = cv2.resize(im, (new_width, new_height))

        # print(f"origin: {width},{height}")
        # print(f"resize: {new_width},{new_height}")
        cv2.imwrite(image_path, im)
        return im

    def convert_box_xywh_to_xyxy(self, box) -> List:
        """
        调整bbox的格式
        """
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def segment_image(self, image, segmentation_mask):
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
        transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
        black_image.paste(segmented_image, mask=transparency_mask_image)

        return black_image

    def get_indices_of_values_above_threshold(self, values, threshold) -> List:
        """
        使用阈值过滤置信度低的分类结果
        """
        indices = [-1] * values.shape[0]
        for i, n in enumerate(values):
            indice = torch.argmax(n)
            if n[indice] > threshold:
                indices[i] = indice
        return indices

    def overlay_image(self, image_path, indices, masks) -> None:
        """
        将mask不同的类别绘制成不同的颜色
        """
        original_image = Image.open(image_path)
        overlay_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
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
            segmentation_mask_image = Image.fromarray(
                masks[seg_idx]["segmentation"].astype("uint8") * 255
            )
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color_map[value])

        result_image = Image.alpha_composite(
            original_image.convert("RGBA"), overlay_image
        )

    def calculate_overlap_ratio(self, image1, image2):

        # 计算交集
        intersection = np.logical_and(image1, image2)
        intersection_area = np.sum(intersection)

        # 计算两个图像的总面积
        total_area_image1 = np.sum(image1)
        total_area_image2 = np.sum(image2)

        # 计算重叠区域比例
        overlap_ratio = intersection_area / min(total_area_image1, total_area_image2)

        return overlap_ratio
    
    def filter_overlap_image(self, masks):
        filter = []
        filter_pairs = []
        overlap_ratios = []
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                
                image1 = masks[i]["segmentation"]
                image2 = masks[j]["segmentation"]
                
                # 将重叠的小的部分都过滤掉
                if np.sum(image1) < np.sum(image2):
                    selected_idx = i
                else:
                    selected_idx = j
                    
                overlap_ratio = self.calculate_overlap_ratio(image1, image2)
                if overlap_ratio >= 0.8:
                    filter.append(selected_idx)
                    filter_pairs.append((i, j))
                    overlap_ratios.append(overlap_ratio)

        # 打印筛选出的图像对
        for idx, pair in enumerate(filter_pairs):
            print(f"Image {pair[0]} and Image {pair[1]} have an overlap ratio of {overlap_ratios[idx]}")

        return filter

    def filter_background(self, masks):
        # filter background
        filter = []
        for i in range(len(masks)):    
            im = masks[i]["segmentation"]
            top_left = im[0:20, 0:20]
            top_right = im[0:20, -20:]
            bottom_left = im[-20:, :20]
            bottom_right = im[-20:, -20:]
            background = np.sum(top_left) + np.sum(top_right) + np.sum(bottom_left) + np.sum(bottom_right)
        #     print(background)
        #     if background >= 720:
            if background >= 360:
                filter.append(i)
        return filter
        

    def detect(self, image_path):
        """
        检测整个场景，输入整张图像，输出该场景中符合数据集类别的信息，包括（中心点，边界框，类别，分割图）
        """

        device = self.device

        # Generate masks
        # self.process_img(image_path)
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)

        # Cut out all masks and each mask's bboxes
        image = Image.open(image_path)
        cropped_imgs, cropped_bboxes = [], []
        for idx, mask in enumerate(masks):
            cropped_imgs.append(
                self.segment_image(image, mask["segmentation"]).crop(
                    self.convert_box_xywh_to_xyxy(mask["bbox"])
                )
            )
            cropped_bboxes.append(mask["bbox"])

            # Visualize bboxes
            # ori_image = cv2.imread("1.jpg")
            # x1, y1, x2, y2 = convert_box_xywh_to_xyxy(mask["bbox"])
            # cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
            # cv2.imwrite("1_.jpg", ori_image)

        # Filter images, 
        filter_idx = []
        filter_idx.extend(self.filter_overlap_image(masks))
        filter_idx.extend(self.filter_background(masks))

        # Use clip to recognize
        @torch.no_grad()
        def retriev(elements, search_label: str) -> int:
            # process imgs
            preprocessed_images = [self.preprocess(image).to(device) for image in elements]
            stacked_images = torch.stack(preprocessed_images)
            image_features = self.model.encode_image(stacked_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # process labels
            label_features = []
            for label in search_label:
                preprocessed_label = torch.stack([self.preprocess(l).to(device) for l in label])
                label_feature = self.model.encode_image(preprocessed_label)
                label_feature /= label_feature.norm(dim=-1, keepdim=True)
                label_feature = torch.mean(label_feature, dim=0, keepdim=False)
                label_features.append(label_feature)
            label_features = torch.stack(label_features)

            # Calculate similar scores
            # print(f"image_shape:{image_features.shape}")
            # print(f"text_shape:{label_features.shape}")
            probs = 100. * image_features @ label_features.T
            print(f"image_text_shape: {probs.shape}")
            return probs[:, :].softmax(dim=-1)

        # Load labeled images
        cls_names = os.listdir(self.label_dir)
        labels = []
        for name in cls_names:
            label = []
            path = os.path.join(self.label_dir, name)
            imgs = os.listdir(path)
            for img in imgs:
                label.append(Image.open(os.path.join(path, img)))
            labels.append(label)

        # Calculate labels
        scores = retriev(cropped_imgs, labels)
        indices = self.get_indices_of_values_above_threshold(scores, 0.1)

        # Process mask, class, center, bbox
        results = {}
        masks_, classes_, centers_, bboxes_ = [], [], [], []
        for seg_idx, cls_idx in enumerate(indices):
            if cls_idx == -1:
                continue
            if seg_idx in filter_idx:
                continue

            if cls_names[cls_idx] == "framework":
                x, y, w, h = cropped_bboxes[seg_idx]
                top_left = [x, y]
                top_right = [x + w, y]
                bottom_left = [x, y + h]
                bottom_right = [x + w, y + h]
                box_center = [x + w / 2, y + w / 2]
                box_right = [x + w, y + h / 2]

                masks_.extend([masks[seg_idx]["segmentation"]] * 6)
                classes_.extend(
                    [
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                        "global_center",
                        "global_center",
                    ]
                )
                centers_.extend(
                    [
                        top_left,
                        top_right,
                        bottom_left,
                        bottom_right,
                        box_center,
                        box_right,
                    ]
                )
                bboxes_.extend([cropped_bboxes[seg_idx]] * 6)

            elif cls_names[cls_idx] == "hand":
                # for hand, get the finger top
                selected_pixel = None
                bin_image = masks[seg_idx]["segmentation"]
                h, w = bin_image.shape
                for i in range(h):
                    if selected_pixel:
                        break
                    if np.sum(bin_image[i]) == 0:
                        continue
                    for j in range(w):
                        if bin_image[i][j] == True:
                            selected_pixel = [j, i]
                            break

                # Visualize finger top point
                # img = cv2.imread('assets/assembly/1.jpg')
                # cv2.circle(img, selected_pixel, 5, [0, 0, 255], -1)
                # cv2.imwrite("1_.png", img)

                masks_.append(masks[seg_idx]["segmentation"])
                classes_.append(cls_names[cls_idx])
                centers_.append(selected_pixel)
                bboxes_.append(cropped_bboxes[seg_idx])

            else:
                # for other classes
                masks_.append(masks[seg_idx]["segmentation"])
                classes_.append(cls_names[cls_idx])
                centers_.append(cropped_bboxes[seg_idx][:2])
                bboxes_.append(cropped_bboxes[seg_idx])

        results["masks"] = np.array(masks_)
        results["classes"] = np.array(classes_)
        results["centers"] = np.array(centers_)
        results["bboxes"] = np.array(bboxes_)

        np.save("info.npy", results)

        return results


if __name__ == "__main__":
    opendetector = OpenDetector("dataset/labels")
    opendetector.detect("1.png")

import os
import sys
sys.path.insert(0, "/workspaces/assemblyhelper")

import numpy as np
import cv2
import math
from utils.detect import detect
from typing import Any, Dict, List, Optional, Tuple

#   ----------------------
#   Gobal variables
#   ----------------------

TOOLS = ["phillips screwdriver", "slotted screwdriver", "hex screwdriver"]
PARTS = ["framework", "stringer", "battery", "signal interface board"]
PICKED_OBJ = None


#   ----------------------
#   Task independence APIs
#   ----------------------

def stop():
    """
    停止机器人运动
    """
    pass


def open_gripper():
    """
    打开夹爪
    """
    pass


def close_gripper():
    """
    关闭夹爪
    """
    pass


def set_speed(rate):
    """
    设置机器人速度
    """
    pass


def move_to_location(pose):
    """
    机器人移动到指定位姿
    """
    # if pose in part space
    # if pose in tool space
    pass


def get_corrent_location():
    """
    获取当前机器人位置
    """
    pass


def get_scene_descriptions(image_path) -> Dict[str, List[Any]]:
    """
    获取场景描述, 返回list形式的场景信息
    """
    results = detect(image_path)
    return results


def get_grasp_pose(scene_des, obj_name) -> List:
    """
    使用视觉获取机器人夹取零件/工具时的位姿
    """
    # set gobal variable pick objs
    PICKED_OBJ = obj_name
    
    grasp_idx = [i for i, v in enumerate(scene_des["classes"]) if v == obj_name][0]
    mask = scene_des["masks"][grasp_idx].astype('uint8') * 255
    
    if obj_name == "crossscrew":
        pose = get_crossscrew_grasp(mask)
        
    print(pose)
    return pose


def get_pointed_assembly_location(scene_des: List) -> List:
    """
    获取手指向的物体装配位姿
    """
    img_path = ""
    pass



#   --------------------
#   Task dependence APIs
#   --------------------

def get_crossscrew_grasp(bin_image):
    """
    获取crossscrew的抓取姿态
    以crossscrew垂直于图像时与Y轴的角度为0, 向左偏为负
    """
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    main_bbox = None

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        area = size[0] * size[1]
        if area > max_area:
            max_area = area
            main_bbox = rect
    # print("Bounding box: ", main_bbox)
    
    (cx, cy), (w, h), r = main_bbox
    if w < h:
        r_rad = r * (math.pi / 180)
        x = cx - np.sin(r_rad) * h * 0.3
        y = cy + np.cos(r_rad) * h * 0.3
    else:
        r_rad = (r-90) * (math.pi / 180)
        x = cx - np.sin(r_rad) * w * 0.3
        y = cy + np.cos(r_rad) * w * 0.3
        
    x = x.astype(int)
    y = y.astype(int)
    
    return (x, y, r_rad)    
    
    

#   ---------------------------------------------
#   Task independence APIs, but not expose to LLM
#   ---------------------------------------------

def pixel2cam(obj_des: List) -> List:
    """
    将像素坐标转化为相机坐标
    """
    pass


def cam2world(obj_pos: List) -> List:
    """
    将相机坐标转化为世界坐标
    """
    pass





if __name__ == '__main__':
    obj_name = ""
    image_path = ""
    scene_des = get_scene_descriptions(image_path)
    target_pose = get_grasp_pose(scene_des, obj_name)
    print(target_pose)

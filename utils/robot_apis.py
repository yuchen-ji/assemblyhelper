import os
import sys
sys.path.insert(0, "/workspaces/assemblyhelper")

import numpy as np
import cv2
import math
from utils.detect import detect


#   --------------------
#   Task independence APIs
#   --------------------

def stop():
    """
    """
    pass

def open_gripper():
    """
    """
    pass

def close_gripper():
    """
    """
    pass

def set_speed(rate):
    """
    """
    pass

def move_to_location(pose):
    """
    """
    # if pose in part space
    # if pose in tool space
    pass

def get_corrent_location():
    """
    """
    pass

def get_tool_storage_location(obj_name):
    """
    """
    move_to_location("tool_space")
    
    # set gobal variable pick objs
    pick_obj = obj_name
    pass

def get_part_storage_location(obj_name):
    """
    """
    move_to_location("part_space")
    
    # set gobal variable pick objs
    pick_obj = obj_name
    pass

def get_pointed_assembly_location():
    """
    """
    img_path = ""
    pass


def get_grasp_pose(obj_name):
    """
    用于获取机器人抓取物体时的夹爪姿态
    """
    results = detect("VM/labels/crossscrew/cross_1.png")
    grasp_idx = [i for i, v in enumerate(results["classes"]) if v == obj_name][0]
    mask = results["masks"][grasp_idx].astype('uint8') * 255
    
    if obj_name == "crossscrew":
        pose = get_crossscrew_grasp(mask)
    
    print(pose)


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
    main_direction = 0
    main_bbox = None

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        area = size[0] * size[1]
        if area > max_area:
            max_area = area
            main_direction = angle
            main_bbox = rect            
    # print("Rotation: ", main_direction)
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








if __name__ == '__main__':
    get_grasp_pose("crossscrew")
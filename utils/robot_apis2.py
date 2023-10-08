import os
import sys
sys.path.insert(0, "/workspaces/assemblyhelper")

import numpy as np
import cv2
import math
from utils.detect import detect
from typing import Any, Dict, List, Optional, Tuple

#   ----------------------
#   Global variables
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
    
    if obj_name == "phillips screwdriver":
        pose = get_phillipsscrew_grasp(mask)
        
    print(pose)
    return pose


def get_pointed_assembly_location(scene_des: List) -> List:
    """
    获取手指向的物体装配位姿
    """
    pass



#   --------------------
#   Task dependence APIs
#   --------------------

def get_phillipsscrew_grasp(bin_image) -> List:
    """
    获取crossscrew的抓取姿态
    以crossscrew垂直于图像时与Y轴的角度为0, 向左偏为负
    """
    rot_bbox = get_rot_bbox(bin_image)
    (cx, cy), (w, h), r = rot_bbox
    
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
    
    return [x, y, r_rad]


def get_slottedscrewdriver_grasp(bin_image) -> List:
    """
    获取一字螺丝刀的抓取姿态
    """
    return get_phillipsscrew_grasp(bin_image)


def get_hexscrewdriver_grasp(bin_image) -> List:
    """
    获取六角螺丝刀的抓取姿态
    """
    return get_phillipsscrew_grasp(bin_image)
    

def get_stringer_grasp(bin_image) -> List:
    """
    获取桁条的抓取姿态
    """
    rot_bbox = get_rot_bbox(bin_image)
    (cx, cy), (w, h), r = rot_bbox
    
    x = cx.astype(int)
    y = cy.astype(int)
    r_rad = 90 * (math.pi / 180)
    
    return [x, y, r_rad]


def get_battery_grasp(bin_image) -> List:
    """
    获取电池的抓取姿态
    """
    return get_stringer_grasp(bin_image)


def get_signalinterfaceboard_grasp(bin_image) -> List:
    """
    获取信号转接板的抓取姿态
    """
    return get_stringer_grasp(bin_image)


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


def get_rot_bbox(bin_image) -> List:
    """
    获取分割图像中，最小的旋转边界框
    """
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    rot_bbox = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        area = size[0] * size[1]
        if area > max_area:
            max_area = area
            rot_bbox = rect
    # print("Bounding box: ", rot_bbox)
    return rot_bbox



if __name__ == '__main__':
    obj_name = "phillips screwdriver"
    image_path = ""
    scene_des = get_scene_descriptions(image_path)
    target_pose = get_grasp_pose(scene_des, obj_name)
    print(target_pose)

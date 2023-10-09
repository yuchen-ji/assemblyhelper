import os
import sys
sys.path.insert(0, "/workspaces/assemblyhelper")

import cv2
import math
import yaml
import numpy as np
from utils.detect import detect
from typing import Any, Dict, List, Optional, Tuple

#   ----------------------
#   Global variables
#   ----------------------

TARGET_OBJ = None
TOOLS = ["phillips screwdriver", "slotted screwdriver", "hex screwdriver"]
PARTS = ["framework", "stringer", "battery", "signal interface board"]
ROBOT_SENSOR = ["deliver_space", "closed", "None"]
ASSEMBLY_LOCATION = None
WORKSPACE = None


#   ----------------------
#   Task independence APIs
#   ----------------------

def stop() -> None:
    """
    停止机器人运动
    """
    pass


def open_gripper() -> None:
    """
    打开夹爪
    """  
    # Set gripper is open and grasped obj is None
    ROBOT_SENSOR[1] = "open"
    ROBOT_SENSOR[2] = "None"
    pass


def close_gripper() -> None:
    """
    关闭夹爪
    """ 
    # Set gripper is closed and grasped obj is target obj
    ROBOT_SENSOR[1] = "closed"
    ROBOT_SENSOR[2] = TARGET_OBJ
    
    # Set some postprocess to avoid collisions
    if ROBOT_SENSOR[1] == "tool_space":
        pose = get_corrent_location()
        pose[2] += 50
        move_to_location(pose)
    
    if ROBOT_SENSOR[2] == "part_space":
        pose = get_corrent_location()
        pose[1] += 50
        move_to_location(pose)
        
    if ROBOT_SENSOR[1] == "assembly_space":
        pose = get_corrent_location()
        pose[2] += 50
        move_to_location(pose)
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
    if isinstance(pose, list):
        pass
    if isinstance(pose, str):
        pose = ASSEMBLY_LOCATION[pose]
        pass
    
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
    # Set global variable target obj
    global TARGET_OBJ
    TARGET_OBJ = obj_name
    
    grasp_idx = [i for i, v in enumerate(scene_des["classes"]) if v == obj_name][0]
    mask = scene_des["masks"][grasp_idx].astype('uint8') * 255
    
    if obj_name == "phillips screwdriver":
        pose = get_phillipsscrew_grasp(mask)
        
    if obj_name == "slotted screwdriver":
        pose = get_slottedscrewdriver_grasp(mask)
        
    if obj_name == "hex screwdriver":
        pose = get_hexscrewdriver_grasp(mask)
        
    if obj_name == "stringer":
        pose = get_stringer_grasp(mask)
        
    if obj_name == "battery":
        pose = get_battery_grasp(mask)
        
    if obj_name == "signal interface board":
        pose = get_signalinterfaceboard_grasp(mask)


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

def get_config() -> None:
    """
    读取配置文件
    """
    with open("utils/config.yml", 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        
    global ASSEMBLY_LOCATION, WORKSPACE
    ASSEMBLY_LOCATION = data["assembly_location"]
    WORKSPACE = data["workspace"]
    

def determine_workspace(pose: list, space: str) -> bool:
    """
    根据输入的pose和所在区域的名称, 判断是否在指定的区域
    """
    status = False
    pose_gt = np.array(ASSEMBLY_LOCATION[space])
    pose = np.array(pose)
    if np.linalg.norm(pose_gt, pose) < 50:
        status = True
    return status


def pixel2cam(obj_des: List) -> List:
    """
    将像素坐标转化为相机坐标
    """
    
    pass


def cam2end(obj_pos: List) -> List:
    """
    将相机坐标转化为机器人末端执行器坐标
    """
    
    pass


def end2base(obj_pos: List) -> List:
    """
    将末端执行器坐标转化为机器人底座坐标
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

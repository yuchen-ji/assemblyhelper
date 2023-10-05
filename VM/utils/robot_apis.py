import numpy as np
import cv2

def stop():
    pass

def open_gripper():
    pass

def close_gripper():
    pass

def set_speed(rate):
    pass

def move_to_position(pose):
    pass

# NOTE: 这个应该如何设计
def get_scene_observation():
    pass

def get_tool_storage_position():
    move_to_position("tool_space")
    pass

def get_part_storage_position():
    move_to_position("part_space")
    pass

def get_assemble_positon():
    pass

def get_corrent_positon():
    pass




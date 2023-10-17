import os
import sys

sys.path.insert(0, "/workspaces/assemblyhelper")

import cv2
import math
import yaml
import json
import queue
import textwrap
import argparse
import threading
import numpy as np
from assemblyhelper.vis_utils import OpenDetector
from assemblyhelper.llm_utils import CodeGenerator
from assemblyhelper.lang_utils import SpeechRecognizer
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

OPENDETECTOR = None
CODEGENERATOR = None
SPEECHRECOGNIZER = None


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
    if isinstance(pose, str):
        pose = WORKSPACE[pose][0]
    if isinstance(pose, list):
        pass
    if isinstance(pose, List[List]):
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

    Args: image_path
    Returns: masks, classes, centers, bboxes
    """
    results = OPENDETECTOR.detect(image_path)
    return results


def get_grasp_pose(scene_des, obj_name) -> List:
    """
    使用视觉获取机器人夹取零件/工具时的位姿
    """
    # Set global variable target obj
    global TARGET_OBJ
    TARGET_OBJ = obj_name

    grasp_idx = [i for i, v in enumerate(scene_des["classes"]) if v == obj_name][0]
    mask = scene_des["masks"][grasp_idx].astype("uint8") * 255

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


#   --------------------
#   Task dependence APIs
#   --------------------


def get_pointed_assembly_location(scene_des: List):
    """
    使用固定的代码获取装配位姿
    """
    hand_idx = [i for i, v in enumerate(scene_des["classes"]) if v == "hand"][0]
    hand_pixel = scene_des["centers"][hand_idx]

    min_distance = 1e10
    ins_num = len(scene_des["masks"])
    for idx in range(ins_num):
        name = scene_des["classes"][idx]
        if name not in ASSEMBLY_LOCATION:
            continue
        pixel = scene_des["centers"][idx]
        distance = np.linalg.norm(hand_pixel - pixel)
        if distance < min_distance:
            min_distance = distance
            pointed_name = name

    pointed_location = ASSEMBLY_LOCATION[pointed_name]
    # print(pointed_location)
    return pointed_location


def get_pointed_assembly_location2(scene_des: List) -> List:
    """
    使用语言模型获取手指向的物体装配位姿
    """
    query_context = "The following is what you observed in the scene:" + "\n"
    query_context += scene_parser(scene_des)
    query_context += textwrap.dedent(
        """
    What is the 6Dpose of the assembly location I am pointing to in your observation?    

    Your should first output the process of thought. 
    But your final outputs need to meet the following format:
    <output> [x, y, z, rx, ry, rz] </output>
    """
    )

    locgenerator = CodeGenerator(preprompt=query_context, oncecall=False)
    loccontext = locgenerator.get_llm_response()
    print(loccontext)

    # 将语言模型的文本形式的输出，转成numpy
    start_idx = loccontext.find("<output>") + len("<output>")
    end_idx = loccontext.find("</output>")
    data_str = loccontext[start_idx:end_idx]
    data_list = eval(data_str)
    print(data_list)
    return data_list


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
        r_rad = (r - 90) * (math.pi / 180)
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


def init(
    label_dir: str,
    checkpoint: str,
    spmodel: str,
    english: bool,
    energy: int,
    pause: int,
    dynamic_energy: bool,
    wake_word: str,
    llmodel: str,
    prompt: str,
    vmprompt: str,
):
    """
    初始化变量，加载模型
    """
    global OPENDETECTOR, CODEGENERATOR, SPEECHRECOGNIZER
    OPENDETECTOR = OpenDetector(label_dir, checkpoint)
    SPEECHRECOGNIZER = SpeechRecognizer(spmodel, english, energy, pause, dynamic_energy, wake_word)
    # CODEGENERATOR = CodeGenerator(file_path=prompt, model=llmodel)


def load_config(file_path) -> None:
    """
    读取配置文件
    """
    with open(file_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    global ASSEMBLY_LOCATION, WORKSPACE
    ASSEMBLY_LOCATION = data["assembly_location"]
    WORKSPACE = data["workspace"]


def determine_workspace() -> str:
    """
    获取当前机器人所在的区域名称
    """
    workspace = "UNKNOWN"
    pose = np.array([120, 150, 200])
    for key, value in WORKSPACE.items():
        ranges = np.array(value[1])
        status = True
        for idx, v in enumerate(pose):
            if v >= ranges[idx][0] and v < ranges[idx][1]:
                continue
            else:
                status = False
                break
        if status:
            workspace = key
            break
    return workspace


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
    contours, _ = cv2.findContours(
        bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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


def scene_parser(scene_des: List) -> str:
    """
    重新组织场景描述的形式
    """
    ins_num = len(scene_des["classes"])
    scene_des_context = ""
    for idx in range(ins_num):
        category = scene_des["classes"][idx]
        pixel_coords = scene_des["centers"][idx]
        if category in ASSEMBLY_LOCATION:
            sixd_poses = ASSEMBLY_LOCATION[category]
            single_format = f"{category}: {{'pixel coords': {pixel_coords}}}; {{'6D pose': {sixd_poses}}}"
            scene_des_context += single_format + "\n"
        elif category == "hand":
            single_format = f"{category}: {{'pixel coords': {pixel_coords}}}"
            scene_des_context += single_format + "\n"

    # print(scene_des_context)
    return scene_des_context


def transfer_instructions(vis_queue, lang_queue):
    """
    将视觉指令和语言指令，调整为合适的格式
    """
    trans_data = None
    while trans_data == None:
        if not vis_queue.empty():
            trans_data = "Human[action]: {}".format(vis_queue.get())
        elif not lang_queue.empty():
            trans_data = "Human[language]: {}".format(lang_queue.get())

    # Get sensor data
    ROBOT_SENSOR[1] = determine_workspace()
    sensor_data = f'Robot[sensor]: [location: "{ROBOT_SENSOR[0]}"; state: "{ROBOT_SENSOR[1]}"; grasped: "{ROBOT_SENSOR[2]}"]'
    query_data = trans_data + "\n" + sensor_data
    return query_data


def argparser():
    """
    获取输入参数
    """
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument(
        "--config",
        default="/workspaces/assemblyhelper/config/config.yml",
        help="装配位置的配置文件",
    )

    # Detector
    parser.add_argument(
        "--label_dir",
        default="/workspaces/assemblyhelper/datasets/labels",
        help="目标检测的标签文件",
    )
    parser.add_argument(
        "--checkpoint",
        default="/workspaces/assemblyhelper/thirdparty/sam_vit_h_4b8939.pth",
        help="SAM的模型权重",
    )

    # Speech to text
    parser.add_argument("--spmodel", default="base", help="使用的whisper模型权重")
    parser.add_argument("--english", default=False, help="是否限制语言类型为英语")
    parser.add_argument("--energy", default=500, help="固定的用于检测声音的阈值")
    parser.add_argument("--pause", default=1.5, help="间隔用于检测短句")
    parser.add_argument("--dynamic_energy", default=True, help="设置动态能量阈值")
    parser.add_argument("--wake_word", default="hey", help="用于唤醒llm响应的唤醒词")

    # Large language model
    parser.add_argument("--llmodel", default="gpt-3.5-turbo", help="使用的语言模型")
    parser.add_argument(
        "--prompt",
        default="/workspaces/assemblyhelper/config/robot_prompt_update2.yml",
        help="用于代码生成的prompt",
    )
    parser.add_argument(
        "--vmprompt",
        default="/workspaces/assemblyhelper/config/vm_prompt.yml",
        help="人手指向分析的prompt",
    )

    args = parser.parse_args()
    return args


#   ---------------------------------------------
#   Test APIs
#   ---------------------------------------------


def test_configs():
    args = argparser()
    load_config(args.config)
    workspace = determine_workspace()
    print(workspace)


def test_pointed():
    # args = argparser()
    # load_config(args.config)
    # init(args.labeldir, args.prompt, args.llmodel)

    # image_path = "/workspaces/assemblyhelper/assets/assembly/bottom_right_2.jpg"
    # scene_des = get_scene_descriptions(image_path)
    # get_pointed_assembly_location(scene_des)
    pass


def test_pipeline():
    """
    测试流程
    """
    args = argparser()
    load_config(args.config)

    detect_keys = dict(
        label_dir = args.label_dir,
        checkpoint = args.checkpoint,
    )
    speech_keys = dict(
        spmodel = args.spmodel,
        english = args.english,
        energy = args.energy,
        pause = args.pause,
        dynamic_energy = args.dynamic_energy,
        wake_word = args.wake_word,
    )
    llm_keys = dict(
        llmodel = args.llmodel,
        prompt = args.prompt,
        vmprompt = args.vmprompt,
    )
    init(**detect_keys, **speech_keys, **llm_keys)

    # Start all moudles: "Speech-to-text", "Action recognition"
    threads = []
    audio_thread = threading.Thread(target=SPEECHRECOGNIZER.record_audio)
    trans_thread = threading.Thread(target=SPEECHRECOGNIZER.transcribe_audio)
    threads.extend([audio_thread, trans_thread])

    # 启动线程，但并不需要等所有线程终止才进行主线程
    for thd in threads:
        thd.start()

    while True:
        # # Text input
        # query = ""
        # context = input("User: ")
        # while context != "q":
        #     query += context + "\n"
        #     context = input("User: ")

        # Speech input
        vis_queue = queue.Queue()   # 之后使用action的模型进行代替
        query = transfer_instructions(vis_queue, SPEECHRECOGNIZER.text_queue)

        # # Action
        # CODEGENERATOR.get_llm_response(query)


def text_query():
    args = argparser()
    load_config(args.config)
    vis_queue, lang_queue = queue.Queue(), queue.Queue()
    transfer_instructions(vis_queue, lang_queue)


if __name__ == "__main__":
    test_pipeline()
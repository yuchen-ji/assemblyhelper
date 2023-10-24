import pyrealsense2 as rs
import numpy as np
import cv2

def show_colorizer_depth_img():
    colorizer = rs.colorizer()
    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    cv2.imshow('filled depth',colorized_depth)


def convert_pixel_to_cam(pixel, depth_frame):
    # intrinsics = rs.intrinsics()
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth_frame.get_distance(pixel[0], pixel[1]))
    print(point)


def get_nearest_plane(color_image, depth_image):
    min_depth = np.min(depth_image)
    # 创建布尔掩码
    mask = (depth_image == min_depth)
    # 进行连通组件分析
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # 找到最大的连通组件（除了背景）
    max_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    # 提取最小深度值的平面
    min_depth_plane = (labels == max_component_index).astype(np.uint8)
    # 为了将平面显示在彩色图像上，将深度值应用到平面
    min_depth_plane = min_depth_plane * 255
    # 将深度平面与彩色图像叠加
    result_image = cv2.addWeighted(color_image, 1, cv2.cvtColor(min_depth_plane, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.imshow('nearest plane', result_image)



if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    #深度图像向彩色对齐
    align_to_color=rs.align(align_to=rs.stream.color)
 
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)

            # 获取depth和color的图像
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # 获取相机参数
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # show_colorizer_depth_img()
            # pixel = [240, 320]
            pixel = [0, 0]
            convert_pixel_to_cam(pixel, depth_frame)
            # get_nearest_plane(color_image, depth_image)

            # Press esc or 'q' to close the image window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

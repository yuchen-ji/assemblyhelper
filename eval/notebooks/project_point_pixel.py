import cv2
import numpy as np
import pyrealsense2 as rs


# 初始化RealSense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取深度传感器的内参
intrinsics_depth = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
intrinsics_rgb = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 获取深度传感器与彩色传感器的转换矩阵
depth_to_color_extrinsics = pipeline.get_active_profile().get_stream(rs.stream.depth).get_extrinsics_to(pipeline.get_active_profile().get_stream(rs.stream.color))

# 将3d坐标转化为2d
# 直接全局求均值
# point3D = [0.14569302, 0.07130019, 0.40764427]
# point3D = [0.16196686, 0.00720197, 0.399213]
# point3D = [0.14966132, -0.05500609, 0.39737633]
# point3D = [-0.00307539, 0.01794564, 0.36453745]
# point3D = [-0.09435005, 0.00812769, 0.39922825]

# 最大最小求均值
point3D = [0.15516222, -0.05525712, 0.40385655]
point3D = [0.14268705, -0.05431373, 0.40385655]
point3D = [0.15532842, -0.0571089, 0.39737633]

point3D = [-0.00340819,  0.01519118, 0.36453745]
# point3D = rs.rs2_transform_point_to_point(depth_to_color_extrinsics, point3D)
point2D = rs.rs2_project_point_to_pixel(intrinsics_rgb, point3D)

print(point2D)

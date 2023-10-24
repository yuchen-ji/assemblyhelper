import cv2
import numpy as np
import pyrealsense2 as rs

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

# config.enable_stream(rs.stream.depth, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align_to_color=rs.align(align_to=rs.stream.color)

# profile = pipeline.get_active_profile()
# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()
# w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()

decimate_level = 0
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2**decimate_level)

filters = [
    rs.disparity_transform(),
    rs.spatial_filter(),
    rs.temporal_filter(),
    rs.disparity_transform(False),
]

while True:

    success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
    if not success:
        continue
    
    frames = align_to_color.process(frames)
    depth_frame = frames.get_depth_frame().as_video_frame()
    color_frame = frames.first(rs.stream.color).as_video_frame()

    depth_frame = decimate.process(depth_frame)

    # # 点云数据后处理
    # for f in filters:
    #     depth_frame = f.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    color_image = np.asanyarray(color_frame.get_data())

    # 将图像中的点转化为3d点云
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)

    # 获取逐像素的点云坐标
    verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
    # texcoords = np.asarray(points.get_texture_coordinates(2)).reshape(h, w, 2)


    # Press esc or 'q' to close the image window
    cv2.imshow("RGB", color_image)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q") or key == 27:
        cv2.destroyAllWindows()
        break

    capture = {}
    capture["color"] = color_image
    capture["pointcloud"] = verts
    np.save('src/workspaces/capture.npy', capture)

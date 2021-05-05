import numpy as np
from numpy.lib.function_base import angle
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2
from utils import plot_box

from pyntcloud import PyntCloud


pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipe.start(config)

frames = pipe.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

color_init = np.asanyarray(color_frame.get_data())
depth_init = np.asanyarray(depth_frame.get_data())

align_to = rs.stream.color
align = rs.align(align_to)

pc = rs.pointcloud()

while True:
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
    spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
    temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    filtered = dec_filter.process(depth_frame)
    filtered = spat_filter.process(filtered)
    filtered = temp_filter.process(filtered)

    depth_frame = filtered
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Visualize point cloud
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # set positions & ranges
    shaft_position = (0.15, -0.2)
    door_positions = [(0.02, 0.785), (0.05, 1.)]
    x_range = [-2, 2]
    y_range = [-0.5, 3]
    z_range = [-1, 0.5]
    angle_thr_degree = 20
    angle_thr = angle_thr_degree * np.pi / 180

    x_positions = verts[:, 0]
    y_positions = verts[:, 2]
    z_positions = verts[:, 1]
    
    # x_positions *= -1
    # z_positions *= -1

    idx = np.where(((x_range[0] < x_positions) & (x_positions < x_range[1])) &\
                   ((y_range[0] < y_positions) & (y_positions < y_range[1])) &\
                   ((z_range[0] < z_positions) & (z_positions < z_range[1])))[0]
    idx_z = np.where((0.19 < z_positions) & (z_positions < 0.21))[0]
    z_line_x = x_positions[idx_z]
    z_line_y = y_positions[idx_z]
    x_positions = x_positions[idx]
    y_positions = y_positions[idx]

    plt.scatter(x_positions, y_positions, s=0.1, color='silver')
    plt.scatter(shaft_position[0], shaft_position[1], s=10, color='green', label='shaft')
    plt.scatter(0., 0., s=10, color='black', label='camera')
    for door_position in door_positions:
        des_position_x = (door_position[0] - shaft_position[0]) * np.cos(angle_thr)\
                        - (door_position[1] - shaft_position[1]) * np.sin(angle_thr) + shaft_position[0]
        des_position_y = (door_position[0] - shaft_position[0]) * np.sin(angle_thr)\
                        + (door_position[1] - shaft_position[1]) * np.cos(angle_thr) + shaft_position[1]

        plt.scatter(des_position_x, des_position_y, s=10, color='red')
        plt.scatter(door_position[0], door_position[1], s=10, color='blue')
        plot_box((des_position_x, des_position_y), scale=0.1)
        # plt.scatter(z_line_x, z_line_y, s=5, color='black')
    plt.xlim(x_range)
    plt.ylim(y_range)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower left')
    plt.pause(0.05)
    plt.cla()


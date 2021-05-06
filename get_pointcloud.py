import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import argparse

from utils import plot_box, count_points_in_box


parser = argparse.ArgumentParser()
parser.add_argument('--flip', action='store_true')
parser.add_argument('--angle', type=float)
args = parser.parse_args()



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
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # point cloud post process
    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()
    filtered = dec_filter.process(depth_frame)
    filtered = spat_filter.process(filtered)
    filtered = temp_filter.process(filtered)
    depth_frame = filtered

    pc.map_to(color_frame)

    points = pc.calculate(depth_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # set positions & ranges
    shaft_position = (0.15, -0.2)
    door_positions = [(0.02, 0.785, 0.), (0.05, 1., 0.1)]
    x_range = [-2, 2]
    y_range = [-0.5, 3]
    z_range = [-1, 0.5]
    angle_degree = 20
    if args.angle is not None:
        angle_degree = args.angle
    angle = angle_degree * np.pi / 180
    box_width = 0.1
    box_height = 0.1

    x_coords = verts[:, 0]
    y_coords = verts[:, 2]
    z_coords = verts[:, 1]
    
    if args.flip:
        x_coords *= -1
        z_coords *= -1

    idx = np.where(((x_range[0] < x_coords) & (x_coords < x_range[1])) &\
                   ((y_range[0] < y_coords) & (y_coords < y_range[1])) &\
                   ((z_range[0] < z_coords) & (z_coords < z_range[1])))[0]
    idx_z = np.where((0.19 < z_coords) & (z_coords < 0.21))[0]
    x_coords = x_coords[idx]
    y_coords = y_coords[idx]
    z_coords = z_coords[idx]

    plt.scatter(x_coords, y_coords, s=0.1, color='silver')
    plt.scatter(shaft_position[0], shaft_position[1], s=10, color='green', label='shaft')
    plt.scatter(0., 0., s=10, color='black', label='camera')
    num_points = 0
    for door_position in door_positions:
        des_position_x = (door_position[0] - shaft_position[0]) * np.cos(angle)\
                        - (door_position[1] - shaft_position[1]) * np.sin(angle) + shaft_position[0]
        des_position_y = (door_position[0] - shaft_position[0]) * np.sin(angle)\
                        + (door_position[1] - shaft_position[1]) * np.cos(angle) + shaft_position[1]
        des_position_z = door_position[2]

        plt.scatter(des_position_x, des_position_y, s=10, color='red')
        plt.scatter(door_position[0], door_position[1], s=10, color='blue')
        plot_box((des_position_x, des_position_y), box_width, box_height)
        num_points += count_points_in_box(x_coords, y_coords, z_coords,
                                  (des_position_x, des_position_y),
                                  box_width, box_height, des_position_z)
    plt.xlim(x_range)
    plt.ylim(y_range)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower left')
    if num_points > 0:
        plt.xlabel('WARNING!!', fontsize=20)
    plt.pause(0.05)
    plt.cla()


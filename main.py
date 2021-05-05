import pdb

import numpy as np
import pyrealsense2 as rs

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

pc = rs.pointcloud()

while True:
    frames = pipe.wait_for_frames()
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

    points.export_to_ply("1.ply", color_frame);
    cloud = PyntCloud.from_file("1.ply");

    import open3d as o3d

    pcd = o3d.io.read_point_cloud("./1.ply")
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])



    print(verts)

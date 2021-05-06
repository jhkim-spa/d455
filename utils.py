import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_box(point, width, height):
    ax = plt.gca()
    box = patches.Rectangle((point[0] - width / 2,
                             point[1] - height / 2),
                             width=width,
                             height=height,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(box)

def count_points_in_box(x_coords, y_coords, z_coords,
    point, width, height, z_coord, offset=0.02):
    idx = np.where((((point[0] - width / 2) < x_coords) &\
                    ((point[0] + width / 2) > x_coords) &\
                    ((point[1] - height / 2) < y_coords) &\
                    ((point[1] + height / 2) > y_coords) &
                    ((z_coord - offset) < z_coords) &
                    ((z_coord + offset) > z_coords)))[0]
    num_points_in_box = idx.shape[0]
    
    # coloring points in box
    plt.scatter(x_coords[idx], y_coords[idx], s=1, color='lightcoral')

    return num_points_in_box

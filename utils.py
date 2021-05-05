import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_box(point, scale):
    ax = plt.gca()
    box = patches.Rectangle((point[0] - scale/2, point[1] - scale/2),
                            width=scale,
                            height=scale,
                            edgecolor='r',
                            facecolor='none')
    ax.add_patch(box)
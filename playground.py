import numpy as np
# from hgg.trajectory_sampler import select_segment_indices
import matplotlib.pyplot as plt
import os
from playground_info import *
from hgg.sampler_utils import select_segment_indices


def visualize_scatter_alone(ax, a, title):
    x = a[:, 0]
    y = a[:, 1]
    t = np.arange(0, len(x))

    scatter =ax.scatter(x, y,  c=t, cmap='gnuplot2', edgecolor='k')
    cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')

    ax.set_aspect('equal')  # Ensuring equal aspect ratio
    ax.grid(True)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    
    plt.savefig(f"playground/{title}.png")
    if cbar:
        cbar.remove()
    ax.cla()  # Clear the current axes
    # fig.clf()  # Clear the entire figure

def visualize_sampled_trajectories(ax, a, start_ends):
    x = a[:, 0]
    y = a[:, 1]
    t = np.arange(0, len(x))

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta', 'brown', 'black']
    scatter =ax.scatter(x, y,  c=t, cmap='gnuplot2', edgecolor='k')
    cbar = ax.figure.colorbar(scatter, ax=ax, label='Time step')
    for i, (start, end) in enumerate(start_ends):
        scatter =ax.scatter(x, y,  c=t, cmap='gnuplot2', edgecolor='k')

        ax.set_aspect('equal')  # Ensuring equal aspect ratio
        ax.grid(True)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.plot(x[start:end], y[start:end], color="red", linewidth=2)

        plt.savefig(f"playground/{i}.png")
        ax.cla()  # Clear the current axes
    fig.clf()  # Clear the entire figure





os.makedirs('playground', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 10))
a = a80
r = r80
visualize_scatter_alone(ax,s, "before merge")
s, r= merge_close_points_with_dbscan(s, r, .35 ,5)
print(s.shape)
visualize_scatter_alone(ax,s, "after merge")

# start_ends = select_segment_indices(rtg                     = r,
#                                     states                  = a, 
#                                     window_size             = 10,
#                                     base_diff_threshold     = .5,
#                                     var_threshold_proportion= 0.1,
#                                     max_changes             = 2,
#                                     similarity_threshold    = 1,
#                                     step_ratio              = 0.5)


# print(start_ends)
# visualize_sampled_trajectories(ax, a, start_ends)

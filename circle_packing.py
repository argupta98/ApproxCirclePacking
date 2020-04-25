""" Implements Approximate Circle Packing Algorithm. """

import numpy as np
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
import itertools
import argparse


class CirclePacker(object):
    """A class to pack circles. """
    def __init__(self):
        self.fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.circles = {}

    def draw_colored_circles(self, circle_centers, radius, color="none"):
        for center in circle_centers:
            center = tuple(center)
            if center in circle_centers:
                circle = self.circles[center]
            else:
                circle = Circle(center, 1, facecolor='none',
                                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
                self.circles[center] = circle
                self.ax.add_patch(circle)
            circle.set_facecolor(color)
            self.fig.canvas.draw()

    def draw_grid(self, grid_x_start, grid_y_start, grid_size, horizonal_length, vertical_length):
        y_locations = np.linspace(grid_x_start, grid_x_start + vertical_length + 1, grid_size)
        lines_h = self.ax.hlines(y=y_locations, min_x=grid_x_start, max_x=grid_x_start + horizonal_length)

        x_locations = np.linspace(grid_y_start, grid_y_start + horizontal_length + 1, grid_size)
        lines_v = self.ax.hlines(x=x_locations, min_y=grid_y_start, max_y=grid_y_start + vertical_length)

        self.fig.canvas.draw()

    def approx_pack_circles(self, circle_centers, grid_size, radius):
        """ Solves the appoximate circle packing problem. 

        Args:
            circle_centers (np.ndarray): (N, 2) array with the locations of the centers of the N circles.
            grid_size (float): The size of grid cell to use for the approximation.
            radius (float): The circle radius
        
        Returns:
            The indices of the circles that are dijoint.
        """

        draw_colored_circles(circle_centers, radius)
        # Impose Grid of Granularity k randomly
        x_offset = np.random.rand()
        y_offset = np.random.rand()

        # Get a bounding box for the set of circles to impose grid on.
        min_x = circle_centers[:, 0].min() - radius
        min_y = circle_centers[:, 1].min() - radius
        max_x = circle_centers[:, 0].max() + radius
        max_y = circle_centers[:, 1].max() + radius

        grid_x_start = min_x - (x_offset)
        grid_y_start = min_y - (y_offset)

        bins_per_row = int(np.ciel((max_x - grid_x_start) / grid_size))
        bins_per_col = int(np.ciel((max_y - grid_y_start) / grid_size))
        draw_grid(grid_x_start, grid_y_start, grid_size, bins_per_row * grid_size, bins_per_col * grid_size)

        max_circles_per_grid = int(np.floor((grid_size / radius)**2))

        circle_bins = default_dict(list)
        # Sort circles into bins
        for idx, center in enumerate(circle_centers):
            # Find if circle is intersected by a line
            # Along x
            from_grid_x_start = center[0] - grid_x_start
            left_grid_line_dist = from_grid_x_start % grid_size
            right_grid_line_dist = grid_size - from_grid_x_start % grid_size 
            if left_grid_line_dist < radius or right_grid_line_dist < radius:
                draw_colored_circles([center], radius, "red")
                continue
            
            # Along y
            from_grid_y_start = center[1] - grid_y_start
            bottom_grid_line_dist = from_grid_y_start % grid_size
            top_grid_line_dist = grid_size - from_grid_y_start % grid_size 
            if bottom_grid_line_dist < radius or top_grid_line_dist < radius:
                draw_colored_circles([center], radius, "red")
                continue

            # If not, assign a bin #
            top_grid_line_idx = int(np.floor((center[1] - grid_y_start) / grid_size))
            left_grid_line_idx = int(np.floor((center[0] - grid_x_start) / grid_size))
            bin_num = top_grid_line_idx * bins_per_ros + left_grid_line_idx
            circle_bins[bin_num].append(idx)
        
        kept_circles = set()
        # For each bin, enumerate all possible circle combinations and pick the one that is best
        for _, center_idx in circle_bins.items():
            set_found = False
            for set_size in range(min(max_circles_per_grid, len(center_idx)), 0, -1):
                all_len_combos = itertools.combinations(center_idx, set_size)
                for combo_center_idx in all_len_combos:
                    if is_valid(circle_centers[combo_center_idx]):
                        kept_circles.update(combo_center_idx)
                        draw_colored_circles(circle_centers[combo_center_idx], radius, "light_blue")
                        set_found = True
                        break
                if set_found:
                    break

        return list(kept_circles)
                
if __name__ == "__main__":
    plt.ion()
    circle_centers = np.array([[-1, -1], [1, 1], [1, -1], [-1, 1])
    radius = 1 
    gridsize = 10
    packer = CirclePacker()
    packer.approx_pack_circles(circle_centers, grid_size, radius)

    
""" Implements Approximate Circle Packing Algorithm. """

import numpy as np
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
import itertools
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--radius', default=0.5, type=float, help="pick the radius for the circles.")
parser.add_argument('--bin_size', default=4, type=float, help="pick the bin size.")
parser.add_argument('--num_circles', default=100, type=int, help="pick number of circles.")

class CirclePacker(object):
    """A class to pack circles. """
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.circles = {}

    def draw_colored_circles(self, circle_centers, radius, color="none"):
        for center in circle_centers:
            center = tuple(center)
            if center in self.circles:
                circle = self.circles[center]
            else:
                circle = matplotlib.patches.Circle(center, radius, facecolor='none',
                                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
                self.circles[center] = circle
                self.ax.add_patch(circle)
            circle.set_facecolor(color)
            self.fig.canvas.draw()

    def draw_grid(self, grid_x_start, grid_y_start, grid_size, horizontal_length, vertical_length):
        y_locations = np.arange(grid_y_start, grid_y_start + vertical_length + 1, grid_size)
        lines_h = self.ax.hlines(y_locations, grid_x_start, grid_x_start + horizontal_length)

        x_locations = np.arange(grid_x_start, grid_x_start + horizontal_length + 1, grid_size)
        lines_v = self.ax.vlines(x_locations, grid_y_start, grid_y_start + vertical_length)

        self.fig.canvas.draw()

    def is_valid(self, centers, radius):
        # compare every circle to the others
        for i, center in enumerate(centers[:-1]):
            for j, other_center in enumerate(centers[i+1:]):
                if np.linalg.norm(center - other_center) < 2 * radius:
                    return False
        return True
                

    def approx_pack_circles(self, circle_centers, grid_size, radius):
        """ Solves the appoximate circle packing problem. 

        Args:
            circle_centers (np.ndarray): (N, 2) array with the locations of the centers of the N circles.
            grid_size (float): The size of grid cell to use for the approximation.
            radius (float): The circle radius
        
        Returns:
            The indices of the circles that are dijoint.
        """

        # Impose Grid of Granularity k randomly
        x_offset = np.random.rand() * grid_size
        y_offset = np.random.rand() * grid_size

        # Get a bounding box for the set of circles to impose grid on.
        min_x = circle_centers[:, 0].min() - radius
        min_y = circle_centers[:, 1].min() - radius
        max_x = circle_centers[:, 0].max() + radius
        max_y = circle_centers[:, 1].max() + radius

        grid_x_start = min_x - x_offset
        grid_y_start = min_y - y_offset

        bins_per_row = int(np.ceil((max_x - grid_x_start) / grid_size))
        bins_per_col = int(np.ceil((max_y - grid_y_start) / grid_size))

        plt_size = max(bins_per_row * grid_size, bins_per_col * grid_size)
        plt.xlim(grid_x_start, grid_x_start + plt_size)
        plt.ylim(grid_y_start, grid_y_start + plt_size)
        
        self.draw_colored_circles(circle_centers, radius)
        time.sleep(3)
        self.draw_grid(grid_x_start, grid_y_start, grid_size, bins_per_row * grid_size, bins_per_col * grid_size)

        max_circles_per_grid = int(np.floor((grid_size / radius)**2))

        circle_bins = defaultdict(list)
        # Sort circles into bins
        for idx, center in enumerate(circle_centers):
            # Find if circle is intersected by a line
            # Along x
            from_grid_x_start = center[0] - grid_x_start
            left_grid_line_dist = from_grid_x_start % grid_size
            right_grid_line_dist = grid_size - left_grid_line_dist
            if left_grid_line_dist < radius or right_grid_line_dist < radius:
                self.draw_colored_circles([center], radius, "red")
                continue
            
            # Along y
            from_grid_y_start = center[1] - grid_y_start
            bottom_grid_line_dist = from_grid_y_start % grid_size
            top_grid_line_dist = grid_size - bottom_grid_line_dist
            if bottom_grid_line_dist < radius or top_grid_line_dist < radius:
                self.draw_colored_circles([center], radius, "red")
                continue

            # If not, assign a bin #
            top_grid_line_idx = int(np.floor((center[1] - grid_y_start) / grid_size))
            left_grid_line_idx = int(np.floor((center[0] - grid_x_start) / grid_size))
            bin_num = top_grid_line_idx * bins_per_row + left_grid_line_idx
            circle_bins[bin_num].append(idx)
        
        time.sleep(3)
        kept_circles = set()
        # For each bin, enumerate all possible circle combinations and pick the one that is best
        for _, center_idx in circle_bins.items():
            set_found = False
            for set_size in range(min(max_circles_per_grid, len(center_idx)), 0, -1):
                all_len_combos = itertools.combinations(center_idx, set_size)
                for combo_center_idx in all_len_combos:
                    if self.is_valid(circle_centers[np.array(combo_center_idx)], radius):
                        kept_circles.update(combo_center_idx)
                        self.draw_colored_circles(circle_centers[np.array(combo_center_idx)], radius, "blue")
                        time.sleep(0.2)
                        set_found = True
                        break
                if set_found:
                    break
        
        finish = raw_input("Finished with {} circles. Press any key to continue.".format(len(kept_circles)))
        return list(kept_circles)
                
if __name__ == "__main__":
    plt.ion()
    args = parser.parse_args()
    circle_centers = np.random.rand(args.num_circles, 2) * 20
    packer = CirclePacker()
    packer.approx_pack_circles(circle_centers, args.bin_size, args.radius)

    
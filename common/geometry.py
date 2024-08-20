import sys
import pathlib
from math import cos, sin, tan, pi
import copy
import numpy as np

def get_xy_matrix(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    points_tuples = [xs, ys]

    return np.array(points_tuples)

class Point:
    def __init__(self, x=0.0, y=0.0, heading=0.0, v=0.0, t=0.0, steer=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.v = v
        self.t = t
        self.steer = steer


class Box:
    def __init__(self, center_x, center_y, heading, length, width):
        self.center = Point(center_x, center_y, heading)
        self.length = length
        self.width = width
        self.sin_heading = sin(self.center.heading)
        self.cos_heading = cos(self.center.heading)

        self.init_corners(center_x, center_y, heading, length, width)
        self.max_x = max(p.x for p in self.corners)
        self.min_x = min(p.x for p in self.corners)
        self.max_y = max(p.y for p in self.corners)
        self.min_y = min(p.y for p in self.corners)

    def init_corners(self, x, y, heading, length, width):
        corners = np.array(
            [
                [
                    -length / 2,
                    length / 2,
                    length / 2,
                    -length / 2,
                ],
                [width / 2, width / 2, -width / 2, -width / 2],
            ]
        )
        Rot1 = np.array([[cos(heading), sin(heading)], [-sin(heading), cos(heading)]])
        corners = (corners.T.dot(Rot1)).T

        corners[0, :] += x
        corners[1, :] += y
        self.corners = []
        for i in range(4):
            point = Point(corners[0][i], corners[1][i], heading)
            self.corners.append(point)

        return corners

    def has_overlap(self, box):
        if (
            box.max_x < self.min_x
            or box.min_x > self.max_x
            or box.max_y < self.min_y
            or box.min_y > self.max_y
        ):
            return False

        shift_x = box.center.x - self.center.x
        shift_y = box.center.y - self.center.y

        dx1 = self.cos_heading * self.length / 2
        dy1 = self.sin_heading * self.length / 2
        dx2 = self.sin_heading * self.width / 2
        dy2 = -self.cos_heading * self.width / 2
        dx3 = box.cos_heading * box.length / 2
        dy3 = box.sin_heading * box.length / 2
        dx4 = box.sin_heading * box.width / 2
        dy4 = -box.cos_heading * box.width / 2

        return (
            abs(shift_x * self.cos_heading + shift_y * self.sin_heading)
            <= abs(dx3 * self.cos_heading + dy3 * self.sin_heading)
            + abs(dx4 * self.cos_heading + dy4 * self.sin_heading)
            + self.length / 2
            and abs(shift_x * self.sin_heading - shift_y * self.cos_heading)
            <= abs(dx3 * self.sin_heading - dy3 * self.cos_heading)
            + abs(dx4 * self.sin_heading - dy4 * self.cos_heading)
            + self.width / 2
            and abs(shift_x * box.cos_heading + shift_y * box.sin_heading)
            <= abs(dx1 * box.cos_heading + dy1 * box.sin_heading)
            + abs(dx2 * box.cos_heading + dy2 * box.sin_heading)
            + box.length / 2
            and abs(shift_x * box.sin_heading - shift_y * box.cos_heading)
            <= abs(dx1 * box.sin_heading - dy1 * box.cos_heading)
            + abs(dx2 * box.sin_heading - dy2 * box.cos_heading)
            + box.width / 2
        )

    def get_plot_corners(self):
        plot_corners = copy.deepcopy(self.corners)
        plot_corners.append(self.corners[0])
        return plot_corners

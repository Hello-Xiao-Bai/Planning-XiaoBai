import sys
import pathlib
from math import cos, sin, tan, hypot, atan2, fabs, pi
import copy
import numpy as np

EPSILON = 1e-6
MAX_VALUE = sys.maxsize


def get_xy_matrix(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    points_tuples = [xs, ys]

    return np.array(points_tuples)


class Point:
    def __init__(self, x=0.0, y=0.0, heading=0.0, s=0.0, v=0.0, t=0.0, steer=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.s = s
        self.v = v
        self.t = t
        self.steer = steer

    def distance_to(self, x, y):
        return hypot(self.x - x, self.y - y)


class Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def length(self):
        return hypot(self.x, self.y)

    def length_square(self):
        return self.x * self.x + self.y * self.y

    def angle(self):
        return atan2(self.y, self.x)

    def normalize(self):
        l = self.Length()
        if l > EPSILON:
            self.x /= l
            self.y /= l

    def distance_to(self, other):
        return hypot(self.x - other.x, self.y - other.y)

    def cross_prod(self, other):
        return self.x * other.y - self.y * other.x

    def inner_prod(self, other):
        return self.x * other.x + self.y * other.y

    def rotate(self, angle):
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        return Vector(
            self.x * cos_angle - self.y * sin_angle,
            self.x * sin_angle + self.y * cos_angle,
        )

    def self_rotate(self, angle):
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        tmp_x = copy.deepcopy(self.x)
        self.x = self.x * cos_angle - self.y * sin_angle
        self.y = tmp_x * sin_angle + self.y * cos_angle

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, ratio):
        return Vector(self.x * ratio, self.y * ratio)

    def __truediv__(self, ratio):
        if fabs(ratio) > EPSILON:
            return Vector(self.x / ratio, self.y / ratio)
        else:
            raise ValueError("Division by zero or very small number")

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, ratio):
        self.x *= ratio
        self.y *= ratio
        return self

    def __itruediv__(self, ratio):
        if fabs(ratio) > EPSILON:
            self.x /= ratio
            self.y /= ratio
            return self
        else:
            raise ValueError("Division by zero or very small number")

    def __eq__(self, other):
        return fabs(self.x - other.x) < EPSILON and fabs(self.y - other.y) < EPSILON


class LineSegment:
    def __init__(self, start_vec, end_vec):
        self.start = start_vec
        self.end = end_vec
        dx = end_vec.x - start_vec.x
        dy = end_vec.y - start_vec.y
        self.length = hypot(dx, dy)
        self.unit_direction = (
            Vector()
            if self.length < EPSILON
            else Vector(dx / self.length, dy / self.length)
        )
        self.heading = self.unit_direction.angle()

    def distance_to(self, vec):
        if self.length < EPSILON:
            return self.start.distance_to(vec)

        start_to_p = vec - self.start
        proj = start_to_p.inner_prod(self.unit_direction)
        if proj <= 0:
            return start_to_p.length()
        if proj > self.length:
            return self.end.distance_to(vec)

        return self.unit_direction.cross_prod(start_to_p)

    def project_onto_unit(self, vec):
        return self.unit_direction.inner_prod(vec - self.start)

    def product_onto_unit(self, vec):
        return self.unit_direction.cross_prod(vec - self.start)


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
            fabs(shift_x * self.cos_heading + shift_y * self.sin_heading)
            <= fabs(dx3 * self.cos_heading + dy3 * self.sin_heading)
            + fabs(dx4 * self.cos_heading + dy4 * self.sin_heading)
            + self.length / 2
            and fabs(shift_x * self.sin_heading - shift_y * self.cos_heading)
            <= fabs(dx3 * self.sin_heading - dy3 * self.cos_heading)
            + fabs(dx4 * self.sin_heading - dy4 * self.cos_heading)
            + self.width / 2
            and fabs(shift_x * box.cos_heading + shift_y * box.sin_heading)
            <= fabs(dx1 * box.cos_heading + dy1 * box.sin_heading)
            + fabs(dx2 * box.cos_heading + dy2 * box.sin_heading)
            + box.length / 2
            and fabs(shift_x * box.sin_heading - shift_y * box.cos_heading)
            <= fabs(dx1 * box.sin_heading - dy1 * box.cos_heading)
            + fabs(dx2 * box.sin_heading - dy2 * box.cos_heading)
            + box.width / 2
        )

    def get_plot_corners(self):
        plot_corners = copy.deepcopy(self.corners)
        plot_corners.append(self.corners[0])
        return plot_corners

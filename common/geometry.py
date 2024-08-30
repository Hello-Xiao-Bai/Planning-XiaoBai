import sys
import pathlib
from math import cos, sin, tan, hypot, atan2, fabs, pi
import copy
import numpy as np
from typing import List, Tuple

EPSILON = 1e-6
MAX_VALUE = sys.maxsize


def get_xy_matrix(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    points_tuples = [xs, ys]

    return np.array(points_tuples)


class Point:
    def __init__(self, x=0.0, y=0.0, theta=0.0, s=0.0, v=0.0, t=0.0, steer=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.s = s
        self.v = v
        self.t = t
        self.steer = steer

    def distance_to(self, x, y):
        return hypot(self.x - x, self.y - y)


class FrenetFramePoint:
    def __init__(self, s=0.0, l=0.0, dl=0.0, ddl=0.0):
        self.s = s
        self.l = l
        self.dl = dl
        self.ddl = ddl


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
    def __init__(self, start_vec: Vector, end_vec: Vector):
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

    def distance_to(self, vec: Vector):
        if self.length < EPSILON:
            return self.start.distance_to(vec)

        start_to_p = vec - self.start
        proj = start_to_p.inner_prod(self.unit_direction)
        if proj <= 0:
            return start_to_p.length()
        if proj > self.length:
            return self.end.distance_to(vec)

        return fabs(self.unit_direction.cross_prod(start_to_p))

    def project_onto_unit(self, vec: Vector):
        return self.unit_direction.inner_prod(vec - self.start)

    def product_onto_unit(self, vec: Vector):
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
        self.corners: list[Point] = []
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


class Polyline:
    def __init__(self, points: List[Point]):
        self.points = points
        self.segments: List[LineSegment] = []
        for i in range(1, len(points)):
            start = Vector(points[i - 1].x, points[i - 1].y)
            end = Vector(points[i].x, points[i].y)
            self.segments.append(LineSegment(start, end))

    def length(self) -> float:
        return self.points[-1].s

    def get_smooth_point(self, s: float):

        if s <= 0.0:
            return copy.deepcopy(self.points[0])

        if s >= self.length():
            return copy.deepcopy(self.points[-1])

        low_index = self.binary_search(s)

        if low_index == len(self.points) - 1:
            return copy.deepcopy(self.points[-1])

        prev_point = self.points[low_index - 1]
        delta_s = s - prev_point.s

        if delta_s < EPSILON:
            return prev_point

        ratio = delta_s / (self.points[low_index].s - prev_point.s)
        interpolation = lambda x1, x2: (x2 - x1) * ratio

        smooth_p = copy.deepcopy(prev_point)
        smooth_p.x += interpolation(prev_point.x, self.points[low_index].x)
        smooth_p.y += interpolation(prev_point.y, self.points[low_index].y)
        smooth_p.heading += interpolation(
            prev_point.heading, self.points[low_index].heading
        )
        smooth_p.s += interpolation(prev_point.s, self.points[low_index].s)
        smooth_p.t += interpolation(prev_point.t, self.points[low_index].t)
        smooth_p.v += interpolation(prev_point.v, self.points[low_index].v)
        smooth_p.steer += interpolation(prev_point.steer, self.points[low_index].steer)

        return smooth_p

    def xy_to_sl(self, x, y):
        min_dis = MAX_VALUE
        vec = Vector(x, y)
        for i in range(len(self.segments)):
            current_min_dis = self.segments[i].distance_to(vec)
            if current_min_dis < min_dis:
                min_index = i
                min_dis = current_min_dis
                nearest_seg = self.segments[i]

        prod = nearest_seg.product_onto_unit(vec)
        proj = nearest_seg.project_onto_unit(vec)

        if min_index == 0:
            accumulate_s = min(proj, nearest_seg.length)
            if proj < 0:
                lateral = prod
            else:
                lateral = min_dis if prod > 0 else -min_dis
        elif min_index == len(self.segments) - 1:
            accumulate_s = self.points[min_index].s + max(0.0, proj)
            if proj > 0:
                lateral = prod
            else:
                lateral = min_dis if prod > 0 else -min_dis
        else:
            accumulate_s = self.points[min_index].s + max(
                0.0, min(proj, nearest_seg.length)
            )
            lateral = prod

        return accumulate_s, lateral

    def sl_to_xy(self, s, l):
        target_p = self.get_smooth_point(s)
        x = target_p.x - sin(target_p.heading) * l
        y = target_p.y + cos(target_p.heading) * l

        return x, y

    def binary_search(self, s: float) -> int:
        left, right = 0, len(self.points) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.points[mid].s < s:
                left = mid + 1
            elif self.points[mid].s > s:
                right = mid - 1
            else:
                return mid

        return left

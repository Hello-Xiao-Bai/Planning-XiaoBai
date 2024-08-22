import sys
import pathlib
from math import cos, sin, tan, pi
import numpy as np

root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))

from common.geometry import *

WHEEL_BASE = 3.0  # 轴距
WIDTH = 2.0  # 车辆宽度
FRONT_TO_WHEEL = 4.0  # 车辆前端到后轴中心距离
BACK_TO_WHEEL = 1.0  # 车辆后端到后轴中心距离
LENGTH = FRONT_TO_WHEEL + BACK_TO_WHEEL  # 车辆长度
MAX_STEER = 0.6  # [rad] 最大转向角


def get_num_str(num, precise=2):
    return f"{num:.{precise}f}"


def normallization(angle):
    return (angle + pi) % (2 * pi) - pi


def xy_to_sl(x, y, points, segments, gif_creator=None):
    min_dis = MAX_VALUE
    vec = Vector(x, y)
    for i in range(len(segments)):
        current_min_dis = segments[i].distance_to(vec)
        if current_min_dis < min_dis:
            min_index = i
            min_dis = current_min_dis
            nearest_seg = segments[i]

    prod = nearest_seg.product_onto_unit(vec)
    proj = nearest_seg.project_onto_unit(vec)
    if gif_creator:
        gif_creator.ax.plot(
            [x, points[min_index].x], [y, points[min_index].y], "purple"
        )

    if min_index == 0:
        accumulate_s = min(proj, nearest_seg.length)
        if proj < 0:
            lateral = prod
        else:
            lateral = min_dis if prod > 0 else -min_dis
    elif min_index == len(segments) - 1:
        accumulate_s = points[min_index].s + max(0.0, proj)
        if proj > 0:
            lateral = prod
        else:
            lateral = min_dis if prod > 0 else -min_dis
    else:
        accumulate_s = points[min_index].s + max(0.0, min(proj, nearest_seg.length))
        lateral = prod

    return accumulate_s, lateral


def sl_to_xy(s_index, l, ref_points, segments, gif_creator=None):
    ref_point = ref_points[s_index]
    x = ref_point.x - sin(ref_point.heading) * l
    y = ref_point.y + cos(ref_point.heading) * l

    if gif_creator:
        gif_creator.ax.plot([x, ref_point.x], [y, ref_point.y], "purple")

    return x, y


def get_car_corners(x, y, theta):
    corners = np.array(
        [
            [
                -BACK_TO_WHEEL,
                (LENGTH - BACK_TO_WHEEL),
                (LENGTH - BACK_TO_WHEEL),
                -BACK_TO_WHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2],
        ]
    )
    Rot1 = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    corners = (corners.T.dot(Rot1)).T

    corners[0, :] += x
    corners[1, :] += y

    return corners

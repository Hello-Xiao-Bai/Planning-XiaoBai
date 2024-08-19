import sys
import pathlib
from math import cos, sin, tan, pi
import numpy as np

root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))


WHEEL_BASE = 3.0  # 轴距
WIDTH = 2.0  # 车辆宽度
FRONT_TO_WHEEL = 4.0  # 车辆前端到后轴中心距离
BACK_TO_WHEEL = 1.0  # 车辆后端到后轴中心距离
LENGTH = FRONT_TO_WHEEL + BACK_TO_WHEEL  # 车辆长度
MAX_STEER = 0.6  # [rad] 最大转向角


def normallization(angle):
    return (angle + pi) % (2 * pi) - pi


def get_car_corners(x, y, theta):
    corners = np.array(
        [
            [
                -BACK_TO_WHEEL,
                (LENGTH - BACK_TO_WHEEL),
                (LENGTH - BACK_TO_WHEEL),
                -BACK_TO_WHEEL,
                -BACK_TO_WHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )
    Rot1 = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    corners = (corners.T.dot(Rot1)).T

    corners[0, :] += x
    corners[1, :] += y

    return corners

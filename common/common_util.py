import sys
import pathlib
from math import cos, sin, tan, pi
import numpy as np
from scipy.spatial.transform import Rotation as Rot

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


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

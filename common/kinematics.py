import numpy as np
from math import cos, sin, tan, pi
import copy
import sys
import pathlib

root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))

from common.common_util import *
from common.geometry import *


def bicycle_model(point, distance):
    next_point = copy.deepcopy(point)
    next_point.x += distance * cos(point.theta)
    next_point.y += distance * sin(point.theta)
    next_point.theta += normallization(distance * tan(point.steer) / WHEEL_BASE)

    return next_point

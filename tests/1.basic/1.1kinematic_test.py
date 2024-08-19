import sys
import pathlib

file_path = pathlib.Path(__file__)
file_name = file_path.stem
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

import copy
from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *


if __name__ == "__main__":
    start_p = Point()
    start_p.theta = pi / 2
    start_p.v = 2.0
    start_p.steer = -0.2
    delta_t = 0.1
    total_t = 20.0

    points = [copy.deepcopy(start_p)]

    for t in np.arange(0, total_t, delta_t):
        new_p = bicycle_model(points[-1], points[-1].v * delta_t)
        points.append(new_p)

    fig = plt.figure()
    plt.title("BicycleModel")
    animation_car(fig, points, save_path=get_gif_path(file_path, str(file_name) + "_1"))

    points = [copy.deepcopy(start_p)]
    for t in np.arange(0, total_t, delta_t):
        new_p = points[-1]
        if t < 3:
            new_p.steer = max(new_p.steer - 0.01, -MAX_STEER)
        else:
            new_p.steer = min(new_p.steer + 0.02, MAX_STEER)
        new_p = bicycle_model(new_p, new_p.v * delta_t)
        points.append(new_p)

    fig = plt.figure()
    plt.title("BicycleModel")
    animation_car(fig, points, save_path=get_gif_path(file_path, str(file_name) + "_2"))

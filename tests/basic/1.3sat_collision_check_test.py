import sys
import pathlib
import random
import copy
from math import cos, sin, tan, pi


file_path = pathlib.Path(__file__)
file_name = file_path.stem
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *


def aabb_and_sat():
    for i in range(0, 10):
        plt.cla()
        x1 = random.uniform(0.0, 10.0)
        y1 = random.uniform(0.0, 10.0)
        theta1 = random.uniform(-pi, pi)
        length1 = random.uniform(2.0, 4.0)
        width1 = random.uniform(1.0, 3.0)
        box1 = Box(x1, y1, theta1, length1, width1)

        x2 = random.uniform(0.0, 10.0)
        y2 = random.uniform(0.0, 10.0)
        theta2 = random.uniform(-pi, pi)
        length2 = random.uniform(2.0, 4.0)
        width2 = random.uniform(1.0, 3.0)
        box2 = Box(x2, y2, theta2, length2, width2)

        has_overlap = box1.has_overlap(box2)

        color = "r" if has_overlap else "k"
        xs, ys = get_xy_matrix(box1.get_plot_corners())
        plt.plot(xs, ys, color=color)
        xs, ys = get_xy_matrix(box2.get_plot_corners())
        plt.plot(xs, ys, color=color)
        plt.title("AABB and SAT Collision Check")
        plt.axis("equal")
        plt.xlim(-5, 15)
        plt.ylim(-5, 15)
        plt.savefig(gif_creator.get_image_path())
        plt.pause(0.7)

    gif_creator.create_gif(0.7)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    aabb_and_sat()

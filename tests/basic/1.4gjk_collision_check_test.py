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
from tests.basic import gjk


def gjk_test():
    fig, ax = plt.subplots()

    for i in range(0, 10):
        plt.cla()
        # test shapes
        x1 = random.uniform(0.0, 10.0)
        y1 = random.uniform(0.0, 10.0)
        theta1 = random.uniform(-pi, pi)
        length1 = random.uniform(2.0, 4.0)
        width1 = random.uniform(1.0, 3.0)
        box1 = Box(x1, y1, theta1, length1, width1)
        shape1 = gjk.Polygon(get_xy_matrix(box1.corners))

        x2 = random.uniform(0.0, 10.0)
        y2 = random.uniform(0.0, 10.0)
        r2 = random.uniform(1.0, 3.0)
        shape2 = gjk.Circle(c=(x2, y2), r=r2)

        # calculate distance and closest points on shapes
        d, v1, v2 = gjk.run_gjk(shape1, shape2)
        print(d)

        # plot shapes and the closest points between them
        color = "r" if d < EPSILON else "k"
        ax.add_patch(
            plt.Polygon(shape1.vertices.T, color=color, fill=False, closed=True)
        )
        ax.add_patch(plt.Circle(shape2.c, shape2.r, edgecolor=color, fill=False))
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], "-o")
        plt.title("GJK Collision Check")
        plt.axis("equal")
        ax.set_xlim([-5, 15])
        ax.set_ylim([-5, 15])
        plt.savefig(gif_creator.get_image_path())
        plt.pause(0.7)

    gif_creator.create_gif(0.7)


if __name__ == "__main__":
    gif_creator = GifCreator(file_path)
    gjk_test()

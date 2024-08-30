import sys
import pathlib
import random
import copy
from math import cos, sin, tan, pi


file_path = pathlib.Path(__file__)
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *


def generate_refline():
    delta_s = 1.0
    points = [Point()]
    relate_vec = Vector(delta_s, 0)
    for i in range(0, 20):
        delta_heading = -pi / 20
        if i < 10:
            delta_heading = -delta_heading
        relate_vec.self_rotate(delta_heading)
        new_point = Point(
            points[-1].x + relate_vec.x,
            points[-1].y + relate_vec.y,
            points[-1].heading + delta_heading,
        )
        new_point.s = points[-1].s + new_point.distance_to(points[-1].x, points[-1].y)
        points.append(new_point)

    return Polyline(points)


def xy_to_sl_test(refline: Polyline):
    ref_xs, ref_ys = get_xy_matrix(refline.points)
    for i in range(0, 5):
        plt.cla()
        plt.plot(ref_xs, ref_ys, "go-", label="ref_points")

        x = random.uniform(0.0, 15.0)
        y = random.uniform(0.0, 15.0)
        s, l = refline.xy_to_sl(x, y)

        plt.plot(x, y, "r*")
        plt.plot(
            [x, refline.get_smooth_point(s).x],
            [y, refline.get_smooth_point(s).y],
            "purple",
        )
        plt.text(
            x - 1, y - 2, "s:" + get_num_str(s) + "\nl:" + get_num_str(l), fontsize=15
        )

        plt.legend()
        plt.title("XY to SL", fontsize=15)
        plt.axis("equal")
        gif_creator.savefig()
        plt.pause(1.5)


def sl_to_xy_test(refline: Polyline):
    ref_xs, ref_ys = get_xy_matrix(refline.points)
    for i in range(0, 5):
        plt.cla()
        plt.plot(ref_xs, ref_ys, "go-", label="ref_points")

        s = random.uniform(0.0, refline.length())
        l = random.uniform(-5, 5.0)
        x, y = refline.sl_to_xy(s, l)

        plt.plot(x, y, "b*")
        plt.plot(
            [x, refline.get_smooth_point(s).x],
            [y, refline.get_smooth_point(s).y],
            "purple",
        )
        plt.text(
            x - 1, y - 2, "s:" + get_num_str(s) + "\nl:" + get_num_str(l), fontsize=15
        )

        plt.legend()
        plt.title("SL to XY", fontsize=15)
        plt.axis("equal")
        gif_creator.savefig()
        plt.pause(1.5)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    refline = generate_refline()
    xy_to_sl_test(refline)
    sl_to_xy_test(refline)
    gif_creator.create_gif(1.5)

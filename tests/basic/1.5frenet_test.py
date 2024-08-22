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
from tests.basic import gjk


def generate_points():
    delta_s = 1.0
    points = [Point()]
    relate_vec = Vector(delta_s, 0)
    segments = []
    for i in range(0, 20):
        delta_heading = -pi / 20
        if i < 10:
            delta_heading = -delta_heading
        relate_vec.self_rotate(delta_heading)
        if i == 0:
            segments.append(LineSegment(Vector(), relate_vec))
        else:
            segments.append(
                LineSegment(segments[-1].end, segments[-1].end + relate_vec)
            )
        new_point = Point(
            points[-1].x + relate_vec.x,
            points[-1].y + relate_vec.y,
            points[-1].heading + delta_heading,
        )
        new_point.s = points[-1].s + new_point.distance_to(points[-1].x, points[-1].y)
        points.append(new_point)

    return points, segments


def xy_to_sl_test(ref_points, segments):
    ref_xs, ref_ys = get_xy_matrix(ref_points)
    for i in range(0, 5):
        plt.cla()
        plt.plot(ref_xs, ref_ys, "go-", label="ref_points")

        x = random.uniform(0.0, 15.0)
        y = random.uniform(0.0, 15.0)
        s, l = xy_to_sl(x, y, ref_points, segments, gif_creator)

        plt.plot(x, y, "r*")
        plt.text(
            x - 1, y - 2, "s:" + get_num_str(s) + "\nl:" + get_num_str(l), fontsize=15
        )

        plt.legend()
        plt.title("XY to SL", fontsize=15)
        plt.axis("equal")
        gif_creator.savefig()
        plt.pause(1.5)


def sl_to_xy_test(ref_points, segments):
    ref_xs, ref_ys = get_xy_matrix(ref_points)
    for i in range(0, 5):
        plt.cla()
        plt.plot(ref_xs, ref_ys, "go-", label="ref_points")

        s_index = random.randrange(0, len(ref_points) - 1)
        s = ref_points[s_index].s
        l = random.uniform(-5, 5.0)
        x, y = sl_to_xy(s_index, l, ref_points, segments, gif_creator)

        plt.plot(x, y, "b*")
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
    ref_points, segments = generate_points()
    xy_to_sl_test(ref_points, segments)
    sl_to_xy_test(ref_points, segments)
    gif_creator.create_gif(1.5)

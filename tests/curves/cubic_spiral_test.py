import sys
import pathlib
import random
import copy
from math import cos, sin, tan, pi
from scipy.integrate import quad


file_path = pathlib.Path(__file__)
file_name = file_path.stem
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *


class CubicSpiral:
    def __init__(self, a0, a1, a2, a3, start_x=0, start_y=0, start_theta=0) -> None:
        self.start_x = start_x
        self.start_y = start_y
        self.start_theta = start_theta
        self.a = [a0, a1, a2, a3]

    def calc_curvature(self, s):
        return self.a[0] + self.a[1] * s + self.a[2] * s**2 + self.a[3] * s**3

    def calc_theta(self, s):
        return normallization(
            self.start_theta
            + (self.a[3] * s**4 / 4)
            + (self.a[2] * s**3 / 3)
            + (self.a[1] * s**2 / 2)
            + (self.a[0] * s)
        )

    def calc_x(self, s):
        def cos_theta(t):
            theta = self.calc_theta(t)
            return cos(theta)

        result, error = quad(cos_theta, 0, s)
        return self.start_x + result

    def calc_y(self, s):
        def sin_theta(t):
            theta = self.calc_theta(t)
            return sin(theta)

        result, error = quad(sin_theta, 0, s)
        return self.start_y + result

    def calc_coarse_x(self, s):
        h = s / 8

        sum_cos = (
            cos(self.calc_theta(0))
            + 4 * cos(self.calc_theta(h))
            + 2 * cos(self.calc_theta(2 * h))
            + 4 * cos(self.calc_theta(3 * h))
            + 2 * cos(self.calc_theta(4 * h))
            + 4 * cos(self.calc_theta(5 * h))
            + 2 * cos(self.calc_theta(6 * h))
            + 4 * cos(self.calc_theta(7 * h))
            + cos(self.calc_theta(s))
        )

        return self.start_x + (s / 24) * sum_cos

    def calc_coarse_y(self, s):
        h = s / 8

        sum_sin = (
            sin(self.calc_theta(0))
            + 4 * sin(self.calc_theta(h))
            + 2 * sin(self.calc_theta(2 * h))
            + 4 * sin(self.calc_theta(3 * h))
            + 2 * sin(self.calc_theta(4 * h))
            + 4 * sin(self.calc_theta(5 * h))
            + 2 * sin(self.calc_theta(6 * h))
            + 4 * sin(self.calc_theta(7 * h))
            + sin(self.calc_theta(s))
        )

        return self.start_y + (s / 24) * sum_sin


def cubic_spiral_test():
    for i in range(0, 10):
        plt.cla()
        a0 = 0
        a1 = random.uniform(-0.1, 0.1)
        a2 = random.uniform(-0.01, 0.01)
        a3 = random.uniform(-0.001, 0.001)
        cubic_spiral = CubicSpiral(a0, a1, a2, a3)
        end_s = 15

        xs, ys = [], []
        coarse_xs, coarse_ys = [], []
        for s in np.arange(0, end_s, 0.01):
            xs.append(cubic_spiral.calc_x(s))
            ys.append(cubic_spiral.calc_y(s))
            coarse_xs.append(cubic_spiral.calc_coarse_x(s))
            coarse_ys.append(cubic_spiral.calc_coarse_y(s))

        plt.plot(xs, ys, color="b", label="quad xy")
        plt.plot(coarse_xs, coarse_ys, color="r", label="Simpson xy")
        plt.title(
            "Cubic Spiral"
            + "\na0:"
            + get_num_str(a0)
            + " a1:"
            + get_num_str(a1)
            + " a2:"
            + get_num_str(a2)
            + " a3:"
            + get_num_str(a3)
        )
        plt.axis("equal")
        plt.legend()
        plt.savefig(gif_creator.get_image_path())
        plt.pause(1.2)

    gif_creator.create_gif(1.2)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    cubic_spiral_test()

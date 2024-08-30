import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import animation
import os

root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))

from common.common_util import *
from common.geometry import *

PAUSE_TIME = 0.001
SAVE_GIF = False

WHEEL_LEN = 0.4  # [m]
WHEEL_WIDTH = 0.1  # [m]
TREAD = 0.7  # [m]


def get_car_plot_points(x, y, theta, steer=0.0):  # pragma: no cover

    fr_wheel = np.array(
        [
            [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
            [
                -WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
            ],
        ]
    )

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    Rot2 = np.array(
        [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
    )

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WHEEL_BASE
    fl_wheel[0, :] += WHEEL_BASE

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    outline = get_car_corners(x, y, theta)
    outline = np.concatenate((outline, outline[:, 0:1]), axis=1)

    return outline, fr_wheel, rr_wheel, fl_wheel, rl_wheel


def plot_car(x, y, theta, steer=0.0, car_color="-k"):
    outline, fr_wheel, rr_wheel, fl_wheel, rl_wheel = get_car_plot_points(
        x, y, theta, steer
    )
    plt.plot(
        np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), car_color
    )
    plt.plot(
        np.array(fr_wheel[0, :]).flatten(),
        np.array(fr_wheel[1, :]).flatten(),
        car_color,
    )
    plt.plot(
        np.array(rr_wheel[0, :]).flatten(),
        np.array(rr_wheel[1, :]).flatten(),
        car_color,
    )
    plt.plot(
        np.array(fl_wheel[0, :]).flatten(),
        np.array(fl_wheel[1, :]).flatten(),
        car_color,
    )
    plt.plot(
        np.array(rl_wheel[0, :]).flatten(),
        np.array(rl_wheel[1, :]).flatten(),
        car_color,
    )
    plt.plot(x, y, "r*")


def get_axes_limits(xs, ys, thetas):
    each_corners = [get_car_corners(x, y, theta) for x, y, theta in zip(xs, ys, thetas)]
    all_corners = np.concatenate(each_corners, axis=1)

    buff = 1.0
    x_min = min(all_corners[0]) - buff
    x_max = max(all_corners[0]) + buff
    y_min = min(all_corners[1]) - buff
    y_max = max(all_corners[1]) + buff

    x_offset = x_max - x_min
    y_offset = y_max - y_min
    if x_offset > y_offset:
        y_min -= x_offset / 2
        y_max += x_offset / 2
    else:
        x_min -= x_offset / 2
        x_max += x_offset / 2

    return x_min, x_max, y_min, y_max


def save_gif(save_path, anim):
    if (save_path != None) & SAVE_GIF:
        save_folder_path = pathlib.Path(save_path).parent
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        anim.save(save_path, fps=30, writer="pillow")


def animation_car(fig, points, car_color="k", traj_color="r", save_path=None):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    thetas = [p.theta for p in points]
    x_min, x_max, y_min, y_max = get_axes_limits(xs, ys, thetas)
    axes = fig.gca()
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    scatter = axes.scatter([], [], c=traj_color)
    lines = [axes.plot([], [], [])[0] for _ in range(6)]

    def update(frame):
        current_frame_point = points[frame]
        for line, data in zip(
            lines,
            get_car_plot_points(
                current_frame_point.x,
                current_frame_point.y,
                current_frame_point.theta,
                current_frame_point.steer,
            ),
        ):
            x, y = data
            line.set_data(x, y)
            line.set_color(car_color)
            scatter.set_offsets(np.c_[current_frame_point.x, current_frame_point.y])

        center_x = [p.x for p in points[: frame + 1]]
        center_y = [p.y for p in points[: frame + 1]]
        lines[5].set_data(center_x, center_y)
        lines[5].set_color(traj_color)

        return lines + [scatter]

    anim = animation.FuncAnimation(
        fig, update, frames=len(points), interval=25, blit=True
    )

    plt.axis("equal")
    plt.show()
    save_gif(save_path, anim)

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import sys
import pathlib

file_path = pathlib.Path(__file__)
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

import copy
from common.geometry import *
from common.plot_util import *
from common.gif_creator import *
from common.common_util import *

# hyper parameter, and they can be adjusted.
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True


# Node class for dijkstra search
class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.cost)
            + ","
            + str(self.parent_index)
        )


def prm_planning(
    start_x,
    start_y,
    goal_x,
    goal_y,
    obstacle_x_list,
    obstacle_y_list,
    robot_radius,
    *,
    rng=None
):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    # Uniform random sampling, and discard points that collide with obstacles.
    sample_x, sample_y = sample_points(
        start_x,
        start_y,
        goal_x,
        goal_y,
        robot_radius,
        obstacle_x_list,
        obstacle_y_list,
        obstacle_kd_tree,
        rng,
    )
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y
    )

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for i, ix, iy in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plt.plot(current.x, current.y, "xg")
            plt.savefig(gif_path.get_image_path())
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id], current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]], [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def construct_env_info():
    ox = []
    oy = []
    border_x = []
    border_y = []

    # road border.
    for i in range(0, 60, 1):
        border_x.append(i)
        border_y.append(0.0)
    for i in range(0, 60, 1):
        border_x.append(60.0)
        border_y.append(i)
    for i in range(0, 61, 1):
        border_x.append(i)
        border_y.append(60.0)
    for i in range(0, 61, 1):
        border_x.append(0.0)
        border_y.append(i)

    # Obstacle 1.
    for i in range(40, 55, 1):
        for j in range(5, 15, 1):
            ox.append(i)
            oy.append(j)

    # Obstacle 2.
    for i in range(40):
        for j in range(20, 25, 1):
            ox.append(j)
            oy.append(i)

    # Obstacle 3.
    for i in range(30):
        for j in range(40, 45, 1):
            ox.append(j)
            oy.append(60.0 - i)
    return border_x, border_y, ox, oy


def prm(rng=None):
    print("Begin to run the prm!!!")

    # start and goal position.
    start_x = 10.0  # [m]
    start_y = 10.0  # [m]
    goal_x = 50.0  # [m]
    goal_y = 50.0  # [m]
    robot_size = 5.0  # [m]

    # construct environment info.
    border_x, border_y, ox, oy = construct_env_info()

    if show_animation:
        plt.plot(border_x, border_y, ".g", markersize=10)
        plt.plot(ox, oy, ".k")
        plt.plot(start_x, start_y, ".r", markersize=20)
        plt.plot(goal_x, goal_y, ".r", markersize=20)
        plt.grid(True)
        plt.axis("equal")

    # run the prm planning.
    ox.extend(border_x)
    oy.extend(border_y)
    rx, ry = prm_planning(start_x, start_y, goal_x, goal_y, ox, oy, robot_size, rng=rng)

    # assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.savefig(gif_path.get_image_path())
        plt.pause(0.001)
        gif_path.create_gif()
        plt.show()


if __name__ == "__main__":
    gif_path = GifCreator(file_path)
    prm()

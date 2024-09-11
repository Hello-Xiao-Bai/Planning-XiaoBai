import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib

file_path = pathlib.Path(__file__)
sys.path.append(str(file_path.parent.parent.parent))
sys.path.append(str(file_path.parent.parent))


import copy
from common.geometry import *
from common.plot_util import *
from common.gif_creator import *
from common.common_util import *
from curves import reeds_shepp_path_test as rs

show_animation = True
show_heuristic_animation = False

############################ Car Info ######################################
WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(
            i_x, i_y, i_yaw, [ox[i] for i in ids], [oy[i] for i in ids]
        ):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # collision

    return True  # no collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(
            x,
            y,
            length * cos(yaw),
            length * sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
            alpha=0.4,
        )
    plt.savefig(gif_creator.get_image_path())


def plot_car(x, y, yaw):
    car_color = "-k"
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw


############################ dynamic programming heuristic #############################
class SimpleNode:

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


def calc_final_path(goal_node, closed_node_set, resolution):
    # generate final course
    rx, ry = [goal_node.x * resolution], [goal_node.y * resolution]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_node_set[parent_index]
        rx.append(n.x * resolution)
        ry.append(n.y * resolution)
        parent_index = n.parent_index

    return rx, ry


def calc_distance_heuristic(gx, gy, ox, oy, resolution, rr):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    resolution: grid resolution [m]
    rr: robot radius[m]
    """

    goal_node = SimpleNode(round(gx / resolution), round(gy / resolution), 0.0, -1)
    ox = [iox / resolution for iox in ox]
    oy = [ioy / resolution for ioy in oy]

    obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = calc_obstacle_map(
        ox, oy, resolution, rr
    )

    motion = get_motion_model()

    open_set, closed_set = dict(), dict()
    open_set[calculate_index(goal_node, x_w, min_x, min_y)] = goal_node
    priority_queue = [(0, calculate_index(goal_node, x_w, min_x, min_y))]

    while True:
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        # show graph
        if show_heuristic_animation:  # pragma: no cover
            plt.plot(current.x * resolution, current.y * resolution, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)

        # Remove the item from the open set

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            node = SimpleNode(
                current.x + motion[i][0],
                current.y + motion[i][1],
                current.cost + motion[i][2],
                c_id,
            )
            n_id = calculate_index(node, x_w, min_x, min_y)

            if n_id in closed_set:
                continue

            if not verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue

            if n_id not in open_set:
                open_set[n_id] = node  # Discover a new node
                heapq.heappush(
                    priority_queue,
                    (node.cost, calculate_index(node, x_w, min_x, min_y)),
                )
            else:
                if open_set[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    open_set[n_id] = node
                    heapq.heappush(
                        priority_queue,
                        (node.cost, calculate_index(node, x_w, min_x, min_y)),
                    )

    return closed_set


def verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False

    if obstacle_map[node.x][node.y]:
        return False

    return True


def calc_obstacle_map(ox, oy, resolution, vr):
    min_x = round(min(ox))
    min_y = round(min(oy))
    max_x = round(max(ox))
    max_y = round(max(oy))

    x_width = round(max_x - min_x)
    y_width = round(max_y - min_y)

    # obstacle map generation
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= vr / resolution:
                    obstacle_map[ix][iy] = True
                    break

    return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width


def calculate_index(node, x_width, x_min, y_min):
    return (node.y - y_min) * x_width + (node.x - x_min)


def get_motion_model():
    # dx, dy, cost
    motion = [
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
        [-1, -1, math.sqrt(2)],
        [-1, 1, math.sqrt(2)],
        [1, -1, math.sqrt(2)],
        [1, 1, math.sqrt(2)],
    ]

    return motion


############################ hybrid a star  ###########################################
XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 20  # number of steer command

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost


class Node:
    def __init__(
        self,
        x_ind,
        y_ind,
        yaw_ind,
        direction,
        x_list,
        y_list,
        yaw_list,
        directions,
        steer=0.0,
        parent_index=None,
        cost=None,
    ):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Path:
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class Config:
    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(-math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kd_tree):
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(
        x_ind,
        y_ind,
        yaw_ind,
        d,
        x_list,
        y_list,
        yaw_list,
        [d],
        parent_index=calc_index(current, config),
        cost=cost,
        steer=steer,
    )

    return node


def is_same_grid(n1, n2):
    if (
        n1.x_index == n2.x_index
        and n1.y_index == n2.y_index
        and n1.yaw_index == n2.yaw_index
    ):
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(
        start_x,
        start_y,
        start_yaw,
        goal_x,
        goal_y,
        goal_yaw,
        max_curvature,
        step_size=MOTION_RESOLUTION,
    )

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analytic_expansion(current, goal, c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
            plt.savefig(gif_creator.get_image_path())
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(
            current.x_index,
            current.y_index,
            current.yaw_index,
            current.direction,
            f_x,
            f_y,
            f_yaw,
            fd,
            cost=f_cost,
            parent_index=f_parent_index,
            steer=f_steer,
        )
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = -MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(
        round(start[0] / xy_resolution),
        round(start[1] / xy_resolution),
        round(start[2] / yaw_resolution),
        True,
        [start[0]],
        [start[1]],
        [start[2]],
        [True],
        cost=0,
    )
    goal_node = Node(
        round(goal[0] / xy_resolution),
        round(goal[1] / xy_resolution),
        round(goal[2] / yaw_resolution),
        True,
        [goal[0]],
        [goal[1]],
        [goal[2]],
        [True],
    )

    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1], ox, oy, xy_resolution, BUBBLE_R
    )

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(
        pq, (calc_cost(start_node, h_dp, config), calc_index(start_node, config))
    )
    final_path = None

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            plt.savefig(gif_creator.get_image_path())
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree
        )

        if is_updated:
            print("path found")
            break

        for neighbor in get_neighbors(current, config, ox, oy, obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if (
                neighbor_index not in openList
                or openList[neighbor_index].cost > neighbor.cost
            ):
                heapq.heappush(pq, (calc_cost(neighbor, h_dp, config), neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = (
        list(reversed(goal_node.x_list)),
        list(reversed(goal_node.y_list)),
        list(reversed(goal_node.yaw_list)),
    )
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):
    ind = (
        (node.yaw_index - c.min_yaw) * c.x_w * c.y_w
        + (node.y_index - c.min_y) * c.x_w
        + (node.x_index - c.min_x)
    )

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


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


def main():
    print("Start Hybrid A* planning")

    # Set Initial parameters
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [50.0, 50.0, np.deg2rad(-90.0)]

    # construct environment info.
    border_x, border_y, ox, oy = construct_env_info()

    if show_animation:
        plt.plot(border_x, border_y, ".k", markersize=10)
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], ".r", markersize=20)
        plt.plot(goal[0], goal[1], ".r", markersize=20)
        plot_arrow(start[0], start[1], start[2], fc="g")
        plot_arrow(goal[0], goal[1], goal[2])
        plt.savefig(gif_creator.get_image_path())
        plt.grid(True)
        plt.axis("equal")

    raw_ox = ox
    raw_oy = oy
    ox.extend(border_x)
    oy.extend(border_y)
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION
    )

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    if show_animation:
        for i, (i_x, i_y, i_yaw) in enumerate(zip(x, y, yaw), start=1):
            if i % 5 == 0:
                plt.cla()
                plt.plot(border_x, border_y, ".k", markersize=10)
                plt.plot(ox, oy, ".k")
                plt.plot(start[0], start[1], ".r", markersize=20)
                plt.plot(goal[0], goal[1], ".b", markersize=20)
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(True)
                plt.axis("equal")
                plot_car(i_x, i_y, i_yaw)
                plt.pause(0.0001)
                plt.savefig(gif_creator.get_image_path())
    gif_creator.create_gif()
    plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()

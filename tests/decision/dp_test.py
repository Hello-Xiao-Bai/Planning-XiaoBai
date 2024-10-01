import math

import matplotlib.pyplot as plt
import sys
import pathlib
from enum import Enum

file_path = pathlib.Path(__file__)
sys.path.append(str(file_path.parent.parent.parent))
sys.path.append(str(file_path.parent.parent))


import copy
from common.geometry import *
from common.plot_util import *
from common.gif_creator import *
from common.common_util import *
from curves.quintic_polynomial_test import QuinticPolynomial

show_animation = True

kLateralDistanceBoundInv = 1.0
kMaxCost = 1e6
kADCLength = 2.0
kADCWidth = 2.0
kDeltaS = 1.0


class Waypoint:
    def __init__(self, point, parent_point=None):
        self.point = point
        self.parent_point = parent_point


class WaypointSampler:
    def __init__(
        self,
        init_point,
        left_boundary,
        right_boundary,
        s_length,
        s_resolution,
        l_resolution,
    ):
        self.init_point = init_point
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.s_length = s_length
        self.s_resolution = s_resolution
        self.l_resolution = l_resolution

    def SampleWaypoint(self):
        waypoints = []
        waypoints.append([Waypoint(self.init_point)])
        num_s_points = int(self.s_length / self.s_resolution) + 1
        for s in range(num_s_points):
            if s == 0:
                continue
            s_val = s * self.s_resolution
            l_points = []
            num_l_points = (
                int((self.left_boundary - self.right_boundary) / self.l_resolution) + 1
            )
            for l in range(num_l_points):
                l_val = self.right_boundary + l * self.l_resolution
                point = FrenetFramePoint(s_val, l_val, 0.0, 0.0)
                l_points.append(Waypoint(point))
            waypoints.append(l_points)
        return waypoints


class Evaluator:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def CalcLateralDistanceCost(self, path):
        lat_dist_abs_sum = 0.0
        lat_dist_sqr_sum = 0.0
        for point in path:
            cost = point.l * kLateralDistanceBoundInv
            lat_dist_sqr_sum += cost * cost
            lat_dist_abs_sum += abs(cost)
        max_abs_sum = max(lat_dist_abs_sum, 0.01)
        return lat_dist_sqr_sum / max_abs_sum

    def CalcCollisionCost(self, path):
        for point in path:
            adc_box = Box(
                point.s,
                point.l,
                point.dl,
                kADCLength,
                kADCWidth,
            )
            if any(obstacle.has_overlap(adc_box) for obstacle in self.obstacles):
                return kMaxCost
        return 1.0

    def CalcTotalCost(self, path):
        lat_dist_cost = self.CalcLateralDistanceCost(path)
        collision_cost = self.CalcCollisionCost(path)
        # TODO: we can add more cost.
        return lat_dist_cost + collision_cost


class ObstacleGenerator:
    def GenerateObstacles(self):
        # TODO: add more obstacles.
        obstacles = []
        obstacles.append(Box(15.0, 0.0, 0.0, 5.0, 2.0))
        obstacles.append(Box(30.0, 5.0, 0.0, 5.0, 2.0))
        obstacles.append(Box(40.0, -5.0, 0.0, 5.0, 2.0))
        obstacles.append(Box(50.0, 0.0, 0.0, 5.0, 4.0))
        return obstacles


class DpRoadGraph:
    def __init__(self, waypoints, obstacles):
        self.waypoints = waypoints
        self.dp_table = [
            [1e9 for j in range(len(waypoints[i]))] for i in range(len(waypoints))
        ]
        self.obstacles = obstacles
        self.evaluator = Evaluator(obstacles)

    def GenerateDpTable(self):
        for i in range(len(self.waypoints)):
            # Represents each col.
            for j in range(len(self.waypoints[i])):
                # Represents each row.
                curr_point = self.waypoints[i][j].point
                if i == 0:
                    self.dp_table[i][j] = 0.0
                else:
                    # Generate all possible paths from previous point to current point.
                    optimal_parent_index = 0
                    for l in range(len(self.waypoints[i - 1])):
                        prev_point = self.waypoints[i - 1][l].point
                        curve = QuinticPolynomial(
                            prev_point.l,
                            prev_point.dl,
                            prev_point.ddl,
                            curr_point.l,
                            curr_point.dl,
                            curr_point.ddl,
                            curr_point.s - prev_point.s,
                        )
                        path = []
                        for s in np.arange(prev_point.s, curr_point.s, kDeltaS):
                            new_l = curve.calc_point(s - prev_point.s)
                            new_dl = curve.calc_first_derivative(s - prev_point.s)
                            new_ddl = curve.calc_second_derivative(s - prev_point.s)
                            new_p = FrenetFramePoint(s, new_l, new_dl, new_ddl)
                            path.append(new_p)
                        # Calculate cost of each path.
                        cost = (
                            self.evaluator.CalcTotalCost(path) + self.dp_table[i - 1][l]
                        )

                        if cost < self.dp_table[i][j]:
                            self.dp_table[i][j] = cost
                            optimal_parent_index = l
                    self.waypoints[i][j].parent_point = self.waypoints[i - 1][
                        optimal_parent_index
                    ]

    def FindOptimalPath(self):
        optimal_point_index = 0
        for i in range(len(self.dp_table[-1])):
            if self.dp_table[-1][i] < self.dp_table[-1][optimal_point_index]:
                optimal_point_index = i
        optimal_point = self.waypoints[-1][optimal_point_index]
        path = [optimal_point.point]
        while optimal_point.parent_point is not None:
            optimal_point = optimal_point.parent_point
            path.append(optimal_point.point)
        path.reverse()

        dense_points = []
        for i in range(len(path) - 1):
            curr_point = path[i]
            next_point = path[i + 1]
            curve = QuinticPolynomial(
                curr_point.l,
                curr_point.dl,
                curr_point.ddl,
                next_point.l,
                next_point.dl,
                next_point.ddl,
                next_point.s - curr_point.s,
            )
            for s in np.arange(curr_point.s, next_point.s, kDeltaS):
                new_l = curve.calc_point(s - curr_point.s)
                new_dl = curve.calc_first_derivative(s - curr_point.s)
                new_ddl = curve.calc_second_derivative(s - curr_point.s)
                new_p = FrenetFramePoint(s, new_l, new_dl, new_ddl)
                dense_points.append(new_p)
        return dense_points


def Plot(obstacles, waypoints, optimal_path_point):
    # Translate to the xy point for visualization.
    optimal_path_point_xy = [
        Point(x=point.s, y=point.l, theta=point.dl) for point in optimal_path_point
    ]
    for i in range(len(optimal_path_point_xy)):
        path_point = optimal_path_point_xy[i]
        plt.cla()
        axes = fig.gca()
        axes.axis("equal")

        # Plot the obstacles.
        for obstacle in obstacles:
            xs, ys = get_xy_matrix(obstacle.get_plot_corners())
            plt.plot(xs, ys, color="k")

        # Plot the waypoints.
        for z in range(len(waypoints)):
            for j in range(len(waypoints[z])):
                point = waypoints[z][j].point
                plt.plot(point.s, point.l, "o", color="gray")

        # Plot the optimal path.
        plt.plot(
            [p.x for p in optimal_path_point_xy[0:i]],
            [p.y for p in optimal_path_point_xy[0:i]],
        )

        plot_car(path_point.x, path_point.y, path_point.theta)
        plt.pause(0.01)
        gif_creator.savefig()
    gif_creator.create_gif()


def main():
    left_boundary = 5
    right_boundary = -5
    s_length = 60.0
    s_resolution = 20.0
    l_resolution = 2.0
    obstacle_generator = ObstacleGenerator()
    obstacles = obstacle_generator.GenerateObstacles()
    init_point = FrenetFramePoint(0.0, 0.0, 0.0, 0.0)
    waypoint_sampler = WaypointSampler(
        init_point, left_boundary, right_boundary, s_length, s_resolution, l_resolution
    )
    waypoints = waypoint_sampler.SampleWaypoint()
    dp_road_graph = DpRoadGraph(waypoints, obstacles)
    dp_road_graph.GenerateDpTable()
    optimal_path_point = dp_road_graph.FindOptimalPath()
    Plot(obstacles, waypoints, optimal_path_point)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()

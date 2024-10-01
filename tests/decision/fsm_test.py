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

kLaneChangeRequestFrame = 20
kLaneChangeUnsafeFrame = 20
kDefaultLaneWidth = 4.0
kInitialL = 0.0


def GenerateQuinticPolynomialPath(start_point, end_point, trajectory_length, delta_s):
    curve = QuinticPolynomial(
        start_point.l,
        start_point.dl,
        start_point.ddl,
        end_point.l,
        end_point.dl,
        end_point.ddl,
        trajectory_length,
    )
    path_points = []
    for s in np.arange(0, trajectory_length + delta_s, delta_s):
        new_l = curve.calc_point(s)
        new_dl = curve.calc_first_derivative(s)
        new_ddl = curve.calc_second_derivative(s)
        new_p = FrenetFramePoint(start_point.s + s, new_l, new_dl, new_ddl)
        path_points.append(new_p)
    return path_points


def GenerateStraightPath(start_point, trajectory_length, delta_s):
    path_points = []
    for s in np.arange(0, trajectory_length + delta_s, delta_s):
        new_l = start_point.l
        new_dl = start_point.dl
        new_ddl = start_point.ddl
        new_p = FrenetFramePoint(start_point.s + s, new_l, new_dl, new_ddl)
        path_points.append(new_p)
    return path_points


class LaneChangeState(Enum):
    LANE_KEEPING = 1
    LANE_CHANGE_LEFT_WAITING = 2
    LANE_CHANGE_RIGHT_WAITING = 3
    LANE_CHANGE_LEFT_RUNNING = 4
    LANE_CHANGE_RIGHT_RUNNING = 5
    LANE_CHANGE_LEFT_CANCEL = 6
    LANE_CHANGE_RIGHT_CANCEL = 7
    LANE_CHANGE_FINISH = 8


class Event(Enum):
    LANE_CHANGE_LEFT_REQUEST = 1
    LANE_CHANGE_RIGHT_REQUEST = 2
    LANE_CHANGE_LEFT_SAFE = 3
    LANE_CHANGE_LEFT_UNSAFE = 4
    LANE_CHANGE_RIGHT_SAFE = 5
    LANE_CHANGE_RIGHT_UNSAFE = 6
    BACK_TO_ORIGINAL_LANE = 7
    NOTHING = 8


class LaneChangeStateMachine:
    def __init__(self):
        self.state = LaneChangeState.LANE_KEEPING

    # @property
    def state(self):
        return self.state

    def handle_event(self, event):
        switch_dict = {
            LaneChangeState.LANE_KEEPING: self.lane_keeping,
            LaneChangeState.LANE_CHANGE_LEFT_WAITING: self.lane_change_left_waiting,
            LaneChangeState.LANE_CHANGE_RIGHT_WAITING: self.lane_change_right_waiting,
            LaneChangeState.LANE_CHANGE_LEFT_RUNNING: self.lane_change_left_running,
            LaneChangeState.LANE_CHANGE_RIGHT_RUNNING: self.lane_change_right_running,
            LaneChangeState.LANE_CHANGE_LEFT_CANCEL: self.lane_change_cancel,
            LaneChangeState.LANE_CHANGE_RIGHT_CANCEL: self.lane_change_cancel,
            LaneChangeState.LANE_CHANGE_FINISH: self.lane_change_finish,
        }
        func = switch_dict.get(self.state)
        func(event)

    def lane_keeping(self, event):
        if event == Event.LANE_CHANGE_LEFT_REQUEST:
            self.state = LaneChangeState.LANE_CHANGE_LEFT_WAITING
        elif event == Event.LANE_CHANGE_RIGHT_REQUEST:
            self.state = LaneChangeState.LANE_CHANGE_RIGHT_WAITING

    def lane_change_left_waiting(self, event):
        if event == Event.LANE_CHANGE_LEFT_SAFE:
            self.state = LaneChangeState.LANE_CHANGE_LEFT_RUNNING

    def lane_change_right_waiting(self, event):
        if event == Event.LANE_CHANGE_RIGHT_SAFE:
            self.state = LaneChangeState.LANE_CHANGE_RIGHT_RUNNING

    def lane_change_left_running(self, event):
        if event == Event.LANE_CHANGE_LEFT_UNSAFE:
            self.state = LaneChangeState.LANE_CHANGE_LEFT_CANCEL

    def lane_change_right_running(self, event):
        if event == Event.LANE_CHANGE_RIGHT_UNSAFE:
            self.state = LaneChangeState.LANE_CHANGE_RIGHT_CANCEL

    def lane_change_cancel(self, event):
        if event == Event.BACK_TO_ORIGINAL_LANE:
            self.state = LaneChangeState.LANE_CHANGE_FINISH

    def lane_change_finish(self, event):
        self.state = LaneChangeState.LANE_KEEPING


def GetLaneChangeRequest(curr_frame):
    # TODO:: implement this function to generate a lane change request
    if curr_frame == kLaneChangeRequestFrame:
        return Event.LANE_CHANGE_LEFT_REQUEST
    else:
        return Event.NOTHING


def GetLaneChangeUnsafeEvent(curr_frame):
    # TODO:: Judge whether the lane change is safe.
    if curr_frame >= kLaneChangeUnsafeFrame:
        return Event.LANE_CHANGE_LEFT_UNSAFE
    else:
        return Event.LANE_CHANGE_LEFT_SAFE


def generate_paths(start_point, trajectory_length, delta_s, lane_change_state_machine):
    # First phase.
    straight_path = GenerateStraightPath(start_point, trajectory_length, delta_s)
    print(f"first phase, lane_change_state: {lane_change_state_machine.state}")

    # Second phase: send a lane change request.
    lane_change_path = []
    final_path = []
    for i in range(len(straight_path)):
        lane_change_request_event = GetLaneChangeRequest(i)
        lane_change_state_machine.handle_event(lane_change_request_event)
        lane_change_state_machine.handle_event(Event.LANE_CHANGE_LEFT_SAFE)
        final_path.append(straight_path[i])
        if lane_change_state_machine.state == LaneChangeState.LANE_CHANGE_LEFT_RUNNING:
            end_point = FrenetFramePoint(
                straight_path[i].s + trajectory_length,
                straight_path[i].l + kDefaultLaneWidth,
                0.0,
                0.0,
            )
            lane_change_path = GenerateQuinticPolynomialPath(
                straight_path[i], end_point, end_point.s - straight_path[i].s, delta_s
            )
            break
    print(f"second phase, lane_change_state: {lane_change_state_machine.state}")

    #  Third phase: send a lane change abort request.
    lane_change_cancel_path = []
    for i in range(len(lane_change_path)):
        lane_change_unsafe_event = GetLaneChangeUnsafeEvent(i)
        lane_change_state_machine.handle_event(lane_change_unsafe_event)
        final_path.append(lane_change_path[i])
        if lane_change_state_machine.state == LaneChangeState.LANE_CHANGE_LEFT_CANCEL:
            end_point = FrenetFramePoint(
                lane_change_path[i].s + trajectory_length, kInitialL, 0.0, 0.0
            )
            lane_change_cancel_path = GenerateQuinticPolynomialPath(
                lane_change_path[i],
                end_point,
                end_point.s - lane_change_path[i].s,
                delta_s,
            )
            break
    print(f"third phase, lane_change_state: {lane_change_state_machine.state}")

    # Fourth phase: back to lane keep.
    for i in range(len(lane_change_cancel_path)):
        final_path.append(lane_change_cancel_path[i])
        if lane_change_cancel_path[i].l < 0.2:
            lane_change_state_machine.handle_event(Event.BACK_TO_ORIGINAL_LANE)
    straight_path_1 = GenerateStraightPath(
        lane_change_cancel_path[len(lane_change_cancel_path) - 1],
        trajectory_length,
        delta_s,
    )
    for point in straight_path_1:
        final_path.append(point)

    print(f"fourth phase, lane_change_state: {lane_change_state_machine.state}")
    return final_path


def plot_paths(final_path, raw_center_line, target_center_line, lane_boundary):
    # Translate to the xy point for visualization.
    final_path_path_xy = [
        Point(x=point.s, y=point.l, theta=point.dl) for point in final_path
    ]
    if show_animation:
        for i in range(len(final_path_path_xy)):
            point = final_path_path_xy[i]
            plt.cla()
            plt.plot(
                [p.x for p in final_path_path_xy[0:i]],
                [p.y for p in final_path_path_xy[0:i]],
            )
            plt.plot(
                [p.s for p in raw_center_line],
                [p.l for p in raw_center_line],
                "g",
            )
            plt.plot(
                [p.s for p in lane_boundary],
                [p.l for p in lane_boundary],
                linestyle="-.",
            )
            plt.plot(
                [p.s for p in target_center_line],
                [p.l for p in target_center_line],
                "g",
            )
            axes = fig.gca()
            axes.set_xlim(0, 70)
            axes.set_ylim(-20.0, 20.0)
            plot_car(point.x, point.y, point.theta)
            plt.pause(0.01)
            gif_creator.savefig()
        gif_creator.create_gif()


def main():
    trajectory_length = 20
    delta_s = 0.5
    start_point = FrenetFramePoint()
    start_point.l = kInitialL
    lane_change_state_machine = LaneChangeStateMachine()

    # Generate paths.
    final_path = generate_paths(
        start_point, trajectory_length, delta_s, lane_change_state_machine
    )

    # Generate center line and lane boundary.
    raw_center_line = GenerateStraightPath(start_point, trajectory_length * 3, delta_s)
    start_point.l = kDefaultLaneWidth
    target_center_line = GenerateStraightPath(
        start_point, trajectory_length * 3, delta_s
    )
    start_point.l = kDefaultLaneWidth * 0.5
    lane_boundary = GenerateStraightPath(start_point, trajectory_length * 3, delta_s)

    # Draw the path.
    plot_paths(final_path, raw_center_line, target_center_line, lane_boundary)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()

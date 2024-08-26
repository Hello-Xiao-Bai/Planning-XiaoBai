import sys
import pathlib

file_path = pathlib.Path(__file__)
file_name = file_path.stem
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(file_path.parent.parent))

import copy
from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *
from curves.quintic_polynomial import QuinticPolynomial


def generate_trajectories(start_p, trajectory_length, max_l, min_l, delta_l, delta_s):
    trajectories = []
    for end_l in np.arange(min_l, max_l + delta_l, delta_l):
        # Generate path with quintic polynomial.
        curve = QuinticPolynomial(
            start_p.l, start_p.dl, start_p.ddl, end_l, 0.0, 0.0, trajectory_length
        )
        points = [start_p]
        for s in np.arange(0, trajectory_length + delta_s, delta_s):
            new_l = curve.calc_point(s)
            new_dl = curve.calc_first_derivative(s)
            new_ddl = curve.calc_second_derivative(s)
            new_p = FrenetFramePoint(start_p.s + s, new_l, new_dl, new_ddl)
            points.append(new_p)
        trajectories.append(points)
    return trajectories


def plot_trajectories(trajectories, color="m"):
    for points in trajectories:
        plt.plot([p.s for p in points], [p.l for p in points], color=color)


def select_optimal_path(paths, random_select=True):
    if random_select:
        return paths[np.random.randint(len(paths))]
    # 这里可以添加更复杂的评估逻辑来选择最佳路径
    return None


def main():
    trajectory_length = 10.0
    max_l = 3.0
    min_l = -3.0
    delta_l = 1.0
    delta_s = 0.2
    start_point = FrenetFramePoint()

    # First phase.
    fig = plt.figure()
    first_phase_trajectories = generate_trajectories(
        start_point, trajectory_length, max_l, min_l, delta_l, delta_s
    )
    plot_trajectories(first_phase_trajectories)

    # Second phase.
    second_phase_paths = []
    for curve in first_phase_trajectories:
        second_phase_trajectories = generate_trajectories(
            curve[-1], trajectory_length, max_l, min_l, delta_l, delta_s
        )
        plot_trajectories(second_phase_trajectories, color="grey")
        for path in second_phase_trajectories:
            # Connect the trajectories of first phase and second phase.
            path = curve[:-1] + path
            second_phase_paths.append(path)

    # Select a optimal path: we can set many cost to evaluate the path, such as:
    # 1. Distance cost.
    # 2. Speed cost.
    # 3. Steer cost.
    # 4. Acceleration cost.
    # 5. Jerk cost.
    # ...
    # Here, we randomly select a path from the curve_list.
    selected_curve = select_optimal_path(second_phase_paths)

    # Translate to the xy point for visualization.
    selected_curve_xy = [
        Point(x=point.s, y=point.l, theta=point.dl) for point in selected_curve
    ]

    plt.title("StateBasedSampler")
    animation_car(
        fig,
        selected_curve_xy,
        save_path=get_gif_path(
            pathlib.Path(__file__), str(pathlib.Path(__file__).stem)
        ),
    )


if __name__ == "__main__":
    main()

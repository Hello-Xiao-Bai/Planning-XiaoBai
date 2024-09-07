import sys
import pathlib

file_path = pathlib.Path(__file__)
file_name = file_path.stem
root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))

import copy
from common.geometry import *
from common.plot_util import *
from common.kinematics import *
from common.common_util import *
from common.gif_creator import *


def generate_trajectories(
    start_p, v, min_steer, max_steer, delta_steer, delta_t, total_t
):
    trajectories = []
    for steer in np.arange(min_steer, max_steer + delta_steer, delta_steer):
        points = [start_p]
        points[-1].steer = steer
        for t in np.arange(0, total_t, delta_t):
            new_p = bicycle_model(points[-1], v * delta_t)
            points.append(new_p)
        trajectories.append(points)
    return trajectories


def plot_trajectories(trajectories, color="m"):
    for points in trajectories:
        plt.plot([p.x for p in points], [p.y for p in points], color=color)


def select_optimal_path(paths, random_select=True):
    if random_select:
        return paths[np.random.randint(len(paths))]
    # 这里可以添加更复杂的评估逻辑来选择最佳路径
    return None


def main():
    start_p = Point()
    start_p.theta = 0.0
    start_p.v = 4.0
    max_steer = 0.6
    min_steer = -0.6
    delta_steer = 0.2
    delta_t = 0.1
    total_t = 2.0

    # First phase.
    fig = plt.figure()
    first_phase_trajectories = generate_trajectories(
        start_p, start_p.v, min_steer, max_steer, delta_steer, delta_t, total_t
    )
    plot_trajectories(first_phase_trajectories)

    # Second phase.
    second_phase_paths = []
    for curve in first_phase_trajectories:
        second_phase_trajectories = generate_trajectories(
            curve[-1], curve[-1].v, min_steer, max_steer, delta_steer, delta_t, total_t
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
    plt.title("ControlBasedSampler")
    animation_car(
        fig,
        selected_curve,
        save_path=get_gif_path(
            pathlib.Path(__file__), str(pathlib.Path(__file__).stem)
        ),
    )


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    main()

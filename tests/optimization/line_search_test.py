import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

file_path = pathlib.Path(__file__)

root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))
from common.gif_creator import *
from common.plot_util import *
from tests.optimization.optimize_method_test import *


def function(x) -> float:
    return x[0] ** 2 + 6 * x[1] ** 2


def function_gradient(x):
    return np.array([x[0] * 2, x[1] * 12])


def armijo(x, d) -> float:
    c1 = 1e-3
    gamma = 0.9
    alpha = 1.0

    while function(x + alpha * d) > function(x) + c1 * alpha * np.dot(
        function_gradient(x).T, d
    ):
        alpha = gamma * alpha
    return alpha


def goldstein(x, d):

    a = 0
    b = np.inf
    alpha = 1
    c1 = 0.1  # 可接受系数
    c2 = 1 - c1
    beta = 2  # 试探点系数

    while np.fabs(a - b) > 1e-5:
        if function(x + alpha * d) <= function(x) + c1 * alpha * np.dot(
            function_gradient(x).T, d
        ):
            if function(x + alpha * d) >= function(x) + c2 * alpha * np.dot(
                function_gradient(x).T, d
            ):
                break
            else:
                a = alpha
                # alpha = (a + b) / 2
                if b < np.inf:
                    alpha = (a + b) / 2
                else:
                    alpha = beta * alpha
        else:
            b = alpha
            alpha = (a + b) / 2

    return alpha


def wolfe(x, d):

    c1 = 0.3
    c2 = 0.9
    alpha = 1
    a = 0
    b = np.inf
    while a < b:
        if function(x + alpha * d) <= function(x) + c1 * alpha * np.dot(
            function_gradient(x).T, d
        ):
            if np.dot(function_gradient(x + alpha * d).T, d) >= c2 * alpha * np.dot(
                function_gradient(x).T, d
            ):
                break
            else:
                a = alpha
                alpha = (a + b) / 2
        else:
            b = alpha
            alpha = (a + b) / 2

    return alpha


def gradient_descent_optimize(x0, line_search, iterations=1000):
    x_i = [x0]
    for i in range(iterations):
        gradient = function_gradient(x_i[i])
        alpha = line_search(x_i[i], -gradient)
        x_i.append(x_i[i] - alpha * gradient)

        if np.linalg.norm(gradient) < 10e-5:
            solution = x_i[i + 1]
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}")
            break
        else:
            solution = None

        print(f"Step {i+1}:{x_i[i+1]}")

    return solution, x_i


def line_search_test():
    x0 = np.array([-5, 8])

    solution, armijo_x_i = gradient_descent_optimize(copy.deepcopy(x0), armijo)
    solution, goldstein_x_i = gradient_descent_optimize(copy.deepcopy(x0), goldstein)
    solution, wolfe_x_i = gradient_descent_optimize(copy.deepcopy(x0), wolfe)

    fig, ax = plt.subplots()
    plot_x_0 = [x_i[0] for x_i in armijo_x_i]
    plot_x_1 = [x_i[1] for x_i in armijo_x_i]
    plt.plot(plot_x_0, plot_x_1, "r*-", label="armijo")

    plot_x_0 = [x_i[0] for x_i in goldstein_x_i]
    plot_x_1 = [x_i[1] for x_i in goldstein_x_i]
    plt.plot(plot_x_0, plot_x_1, "g*-", label="goldstein")

    plot_x_0 = [x_i[0] for x_i in wolfe_x_i]
    plot_x_1 = [x_i[1] for x_i in wolfe_x_i]
    plt.plot(plot_x_0, plot_x_1, "y*-", label="wolfe")
    plt.plot(x0[0], x0[1], "ko", label="x0")

    for i in range(11):
        ax.add_patch(Circle((0, 0), i, facecolor="k", alpha=0.3))

    plt.title("Line Search Test")
    ax.axis("equal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    line_search_test()

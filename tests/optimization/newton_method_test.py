import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pathlib

file_path = pathlib.Path(__file__)

root_dir = file_path.parent.parent.parent
sys.path.append(str(root_dir))
from common.gif_creator import *

from common.plot_util import *


def get_gradient(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],  # Add x0 as argument
) -> np.ndarray:
    """
    Calculate the gradient of a function at a given point.

    Args:
        function (sm.core.expr.Expr): The function to calculate the gradient of.
        symbols (list[sm.core.symbol.Symbol]): The symbols representing the variables in the function.
        x0 (dict[sm.core.symbol.Symbol, float]): The point at which to calculate the gradient.

    Returns:
        numpy.ndarray: The gradient of the function at the given point.
    """
    d1 = {}
    gradient = np.array([])

    for i in symbols:
        d1[i] = sm.diff(function, i, 1).evalf(subs=x0)  # add evalf method
        gradient = np.append(gradient, d1[i])

    return gradient.astype(np.float64)  # Change data type to float


def get_hessian(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
) -> np.ndarray:
    """
    Calculate the Hessian matrix of a function at a given point.

    Args:
    function (sm.core.expr.Expr): The function for which the Hessian matrix is calculated.
    symbols (list[sm.core.symbol.Symbol]): The list of symbols used in the function.
    x0 (dict[sm.core.symbol.Symbol, float]): The point at which the Hessian matrix is evaluated.

    Returns:
    numpy.ndarray: The Hessian matrix of the function at the given point.
    """
    d2 = {}
    hessian = np.array([])

    for i in symbols:
        for j in symbols:
            d2[f"{i}{j}"] = sm.diff(function, i, j).evalf(subs=x0)
            hessian = np.append(hessian, d2[f"{i}{j}"])

    hessian = np.array(np.array_split(hessian, len(symbols)))

    return hessian.astype(np.float64)


def newton_method(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
    iterations: int = 100,
) -> dict[sm.core.symbol.Symbol, float] or None:
    """
    Perform Newton's method to find the solution to the optimization problem.

    Args:
        function (sm.core.expr.Expr): The objective function to be optimized.
        symbols (list[sm.core.symbol.Symbol]): The symbols used in the objective function.
        x0 (dict[sm.core.symbol.Symbol, float]): The initial values for the symbols.
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        dict[sm.core.symbol.Symbol, float] or None: The solution to the optimization problem, or None if no solution is found.
    """

    x_star = {}
    x_star[0] = np.array(list(x0.values()))

    print(f"Starting Values: {x_star[0]}")

    for i in range(iterations):

        gradient = get_gradient(function, symbols, dict(zip(x0.keys(), x_star[i])))
        hessian = get_hessian(function, symbols, dict(zip(x0.keys(), x_star[i])))

        x_star[i + 1] = x_star[i].T - np.linalg.inv(hessian) @ gradient.T

        if np.linalg.norm(x_star[i + 1] - x_star[i]) < 10e-5:
            solution = dict(zip(x0.keys(), x_star[i + 1]))
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}")
            break
        else:
            solution = None

        print(f"Step {i+1}: {x_star[i+1]}")

    return solution, x_star


def rosenbrock_function(x, y):
    return 100 * (y - x**2) ** 2 + (1 - x) ** 2


def newton_method_test():
    x, y = sm.symbols("x y")
    Gamma = [x, y]
    objective = rosenbrock_function(x, y)

    solution, x_star = newton_method(objective, Gamma, {x: -1.2, y: 1})

    for x_i in x_star.values():
        xs, ys, zs = [], [], []
        ax = fig.add_subplot(111, projection="3d")
        xs, ys = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-5, 2, 0.1))
        zs = np.array(
            [rosenbrock_function(x, y) for x, y in zip(np.ravel(xs), np.ravel(ys))]
        ).reshape(xs.shape)

        ax.plot_surface(xs, ys, zs, alpha=0.5, label="Rosenbrock ")
        ax.set_xlabel("X ")
        ax.set_ylabel("Y ")
        ax.set_zlabel("Z ")

        ax.scatter(
            x_i[0],
            x_i[1],
            rosenbrock_function(x_i[0], x_i[1]),
            c="r",
            marker="*",
            s=10,
            label="x_i",
        )
        plt.title("Optimize Rosenbrock Function by Newton Method")
        plt.legend()
        gif_creator.savefig()
        plt.pause(0.5)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    gif_creator = GifCreator(file_path, fig, ax)
    newton_method_test()
    gif_creator.create_gif()

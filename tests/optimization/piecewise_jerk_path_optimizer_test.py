import osqp
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


class PiecewiseJerkPathOptimizer:
    def __init__(self, start_states: []):
        total_length = 100
        self.start_states = start_states
        self.delta_s = 0.1
        self.n = int(total_length / self.delta_s)
        self.upper_bound = np.zeros(self.n)
        self.lower_bound = np.zeros(self.n)
        self.ref_s = np.zeros(self.n)

        self.calc_bound()
        self.calc_P_q()
        self.calc_A()
        self.calc_l_u()

    def calc_bound(self):
        l_bound = 2
        for i in range(self.n):
            if i > 200 and i < 400:
                self.upper_bound[i] = 0
                self.lower_bound[i] = -l_bound
            elif i > 600 and i < 800:
                self.upper_bound[i] = l_bound
                self.lower_bound[i] = 0
            else:
                self.upper_bound[i] = l_bound
                self.lower_bound[i] = -l_bound

            self.ref_s[i] = 0.5 * (self.upper_bound[i] + self.lower_bound[i])

    def calc_P_q(self):
        weight_l = 1
        weight_dl = 100
        weight_ddl = 100
        weight_dddl = 1000
        eye_n = np.identity(self.n)
        zero_n = np.zeros((self.n, self.n))

        P_zeros = zero_n
        P_l = weight_l * eye_n
        P_dl = weight_dl * eye_n
        P_ddl = (
            weight_ddl + 2 * weight_dddl / self.delta_s / self.delta_s
        ) * eye_n - 2 * weight_dddl / self.delta_s / self.delta_s * np.eye(self.n, k=-1)
        P_ddl[0][0] = weight_ddl + weight_dddl / self.delta_s / self.delta_s
        P_ddl[self.n - 1][self.n - 1] = (
            weight_ddl + weight_dddl / self.delta_s / self.delta_s
        )

        self.P = sparse.csc_matrix(
            np.block(
                [
                    [P_l, P_zeros, P_zeros],
                    [P_zeros, P_dl, P_zeros],
                    [P_zeros, P_zeros, P_ddl],
                ]
            )
        )
        self.q = np.zeros(self.n * 3)
        self.q[: self.n] = -weight_l * self.ref_s

    def calc_A(self):
        eye_n = np.identity(self.n)
        zero_n = np.zeros((self.n, self.n))
        A_ll = eye_n - np.eye(self.n, k=1)
        A_ldl = self.delta_s * eye_n
        A_lddl = (1.0 / 3.0) * self.delta_s * self.delta_s * eye_n + (
            1.0 / 6.0
        ) * self.delta_s * self.delta_s * np.eye(self.n, k=1)
        A_l = np.block([[A_ll, A_ldl, A_lddl]])

        A_dll = zero_n
        A_dldl = eye_n - np.eye(self.n, k=1)
        A_dlddl = 0.5 * self.delta_s * (eye_n + np.eye(self.n, k=1))
        A_dl = np.block([[A_dll, A_dldl, A_dlddl]])

        A_ul = np.block(
            [
                [eye_n, zero_n, zero_n],
                [zero_n, zero_n, zero_n],
                [zero_n, zero_n, zero_n],
            ]
        )

        A_init = np.zeros((3, 3 * self.n))
        A_init[0][0] = 1
        A_init[1][self.n] = 1
        A_init[2][self.n * 2] = 1

        self.A = sparse.csc_matrix(np.row_stack((A_ul, A_l, A_dl, A_init)))

    def calc_l_u(self):
        self.l = np.zeros(5 * self.n + 3)
        self.u = np.zeros(5 * self.n + 3)
        self.l[: self.n] = self.lower_bound
        self.u[: self.n] = self.upper_bound
        self.l[self.n : self.n * 2] = -2.0
        self.u[self.n : self.n * 2] = 2.0
        self.l[self.n * 2 : self.n * 3] = -2.0
        self.u[self.n * 2 : self.n * 3] = 2.0

        self.l[-3:] = self.u[-3:] = self.start_states

    def optimize(self):
        solver = osqp.OSQP()
        solver.setup(self.P, self.q, self.A, self.l, self.u)
        return solver.solve()


def piecewise_jerk_path_optimizer_test():
    start_states = [1, -0.2, 0]
    planner = PiecewiseJerkPathOptimizer(start_states)
    result = planner.optimize()

    plt.plot(planner.upper_bound, color="black", label="bound")
    plt.plot(planner.lower_bound, color="black")
    plt.plot(planner.ref_s, "--", color="green", label="ref_line")
    plt.plot(result.x[: planner.n], color="red", label="optimal path")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    piecewise_jerk_path_optimizer_test()

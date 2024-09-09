import osqp
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import block_diag
import time


class QPPathPlanner:
    def __init__(self, horizon):
        self.horizon = horizon
        self.uniform_ds = 0.1
        self.ds = np.ones(self.horizon - 1) * self.uniform_ds
        self.init_l = 0.0
        self.init_dl = 0.0
        self.init_ddl = 0.0
        self.l_ref = np.zeros(self.horizon)
        self.l_weight = 1
        self.dl_weight = 100
        self.ddl_weight = 30000
        self.l_max = np.ones((self.horizon, 1))
        self.l_min = np.ones((self.horizon, 1))
        self.ddl_max = np.ones((self.horizon, 1))
        self.ddl_min = np.ones((self.horizon, 1))
        self.l_res = np.zeros(self.horizon)

    def set_reference_l(self, l_ref):
        self.l_ref = l_ref

    def set_weight(self, l_weight, dl_weight, ddl_weight):
        self.l_weight = l_weight
        self.dl_weight = dl_weight
        self.ddl_weight = ddl_weight

    def set_initial_condition(self, init_l, init_dl, init_ddl):
        self.init_l = init_l
        self.init_dl = init_dl
        self.init_ddl = init_ddl

    def set_l_boundary(self, l_max, l_min):
        self.l_max = np.reshape(l_max, (self.horizon, 1))
        self.l_min = np.reshape(l_min, (self.horizon, 1))

    def set_ddl_boundary(self, ddl_max, ddl_min):
        self.ddl_max = np.ones((self.horizon, 1)) * ddl_max
        self.ddl_min = np.ones((self.horizon, 1)) * ddl_min

    def set_uniform_ds(self, ds):
        self.uniform_ds = ds
        self.ds = np.ones(self.horizon - 1) * self.uniform_ds

    def set_ds(self, ds):
        self.ds = ds

    def get_optimized_l(self):
        return self.l_res

    def compute_P_q(self):
        weight = np.array(
            [
                [self.l_weight, 0.0, 0.0],
                [0.0, self.dl_weight, 0.0],
                [0.0, 0.0, self.ddl_weight],
            ]
        )
        P = sparse.kron(sparse.eye(self.horizon), weight).tocsc()
        q = np.zeros(self.horizon * 3)
        for i in range(self.horizon):
            q[i * 3] = -self.l_weight * self.l_ref[i]
        return P, q

    def compute_A(self):
        Ad = []
        for i in range(self.horizon - 1):
            f_i = np.array([[1.0, self.ds[i], 0.0], [0.0, 1.0, self.ds[i]]])
            Ad.append(f_i)
        Ax = sparse.csr_matrix(block_diag(*Ad))
        f_2 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        Ay = sparse.kron(sparse.eye(self.horizon - 1), f_2)
        off_set = np.zeros(((self.horizon - 1) * 2, 3))
        Ax = sparse.hstack([Ax, off_set])
        Ay = sparse.hstack([off_set, Ay])
        Aeq = Ax + Ay
        ineq_l = np.array([1.0, 0.0, 0.0])
        ineq_ddl = np.array([0.0, 0.0, 1.0])
        Aineq_l = sparse.kron(sparse.eye(self.horizon), ineq_l)
        Aineq_ddl = sparse.kron(sparse.eye(self.horizon), ineq_ddl)
        A_init_l = np.zeros(self.horizon * 3)
        A_init_l[0] = 1
        A_init_dl = np.zeros(self.horizon * 3)
        A_init_dl[1] = 1
        A_init_ddl = np.zeros(self.horizon * 3)
        A_init_ddl[2] = 1

        A = sparse.vstack(
            [Aeq, Aineq_l, Aineq_ddl, A_init_l, A_init_dl, A_init_ddl]
        ).tocsc()
        return A

    def compute_u_l(self):
        ueq = np.zeros(((self.horizon - 1) * 2, 1))
        leq = ueq
        uineq = np.vstack([self.l_max, self.ddl_max])
        lineq = np.vstack([self.l_min, self.ddl_min])
        u_init = np.array([[self.init_l], [self.init_dl], [self.init_ddl]])
        l_init = u_init
        l = np.vstack([leq, lineq, l_init])
        u = np.vstack([ueq, uineq, u_init])
        return u, l

    def optimize(self):
        P, q = self.compute_P_q()
        A = self.compute_A()
        u, l = self.compute_u_l()
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        res = prob.solve()
        for i in range(self.horizon):
            self.l_res[i] = res.x[i * 3]

    def plot(self):
        plt.plot(self.l_res, color="g", label="optimized path")
        plt.plot(self.l_min, color="blue", label="lower bound")
        plt.plot(self.l_max, color="red", label="upper bound")
        plt.legend()
        plt.title("QP Path Planner")
        plt.show()


if __name__ == "__main__":
    horizon = 400
    ds = 0.25
    init_l = 0
    init_dl = 0
    init_ddl = 0
    l_max = []
    l_min = []
    for i in range(horizon):
        u = 5
        if i > 100 and i < 140:
            u = -2
        l_max.append(u)
        l = -5
        if i > 300 and i < 340:
            l = 2
        l_min.append(l)
    ddl_max = 0.2
    ddl_min = -0.2
    planner = QPPathPlanner(horizon)
    planner.set_uniform_ds(ds)
    planner.set_initial_condition(init_l, init_dl, init_ddl)
    planner.set_l_boundary(l_max, l_min)
    planner.set_ddl_boundary(ddl_max, ddl_min)
    planner.optimize()
    planner.plot()

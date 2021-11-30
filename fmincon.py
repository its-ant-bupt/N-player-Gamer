import math
import time

import numpy
from scipy.optimize import minimize
import numpy as np


def gamer(n, Us, p, I, s, x0, pay, U):
    def funct(x):
        Funct = 0
        prod = 1
        for i in range(n):
            Funct = Funct + x[s+i]
        for i in range(p):
            for j in range(n):
                prod = prod * x[I[i][j]]
            Funct = Funct - Us[i] * prod
            prod = 1
        # v = lambda x: Funct
        return Funct

    def fun():
        v = lambda x: funct(x)
        return v

    def confun(x):
        C = np.zeros(s)
        for i in range(s):
            C[i] = x[pay[i]]
            for t in range(n):
                add = 0
                for j in range(p):
                    prd = 1
                    for k in range(n):
                        if i == I[j][k]:
                            prd = prd * U[j][k]
                        else:
                            prd = prd * x[I[j][k]]
                    if I[j][t] != i:
                        prd = 0
                    add = add + prd
                C[i] = C[i] - add
        c = C
        # ceq = []
        return c

    def con():
        x_min = 0.0
        x_max = 1.0
        o_min = -1000
        o_max = 1000

        _cons = (
            # 非线性不等式约束 g(x) <= 0
            {'type': 'ineq', 'fun': lambda x: confun(x)[0]},
            {'type': 'ineq', 'fun': lambda x: confun(x)[1]},
            {'type': 'ineq', 'fun': lambda x: confun(x)[2]},
            {'type': 'ineq', 'fun': lambda x: confun(x)[3]},
            {'type': 'ineq', 'fun': lambda x: confun(x)[4]},
            {'type': 'ineq', 'fun': lambda x: confun(x)[5]},
            # 上下界约束
            {'type': 'ineq', 'fun': lambda x: x[0] - x_min},
            {'type': 'ineq', 'fun': lambda x: x[1] - x_min},
            {'type': 'ineq', 'fun': lambda x: x[2] - x_min},
            {'type': 'ineq', 'fun': lambda x: x[3] - x_min},
            {'type': 'ineq', 'fun': lambda x: x[4] - x_min},
            {'type': 'ineq', 'fun': lambda x: x[5] - x_min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x_max},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x_max},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x_max},
            {'type': 'ineq', 'fun': lambda x: -x[3] + x_max},
            {'type': 'ineq', 'fun': lambda x: -x[4] + x_max},
            {'type': 'ineq', 'fun': lambda x: -x[5] + x_max},
            # 非线性等式约束
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},
            {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 1},
            {'type': 'eq', 'fun': lambda x: x[4] + x[5] - 1}
        )
        return _cons

    cons = con()
    res = minimize(funct, x0, constraints=cons, method="SLSQP")
    return res


if __name__ == '__main__':
    n = 3
    Us = [9.5, 3.5, 11.1, 10.5, 12.2, 10.2, 5.2, 3.1]
    p = 8
    I = [[0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
         [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]]
    s = 6
    pay = [6, 6, 7, 7, 8, 8]
    U = [[2.0, 7.5, 0.0], [3.0, 0.2, 0.3], [6.0, 3.6, 1.5], [2.0, 3.5, 5.0],
         [0.0, 3.2, 9.0], [2.0, 3.2, 5.0], [0.0, 2.0, 3.2], [2.1, 0.0, 1.0]]
    x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2672, 0.3625, 0.3906]
    time1 = time.time()
    res = gamer(n=n, Us=Us, p=p, I=I, s=s, pay=pay, U=U, x0=x0)
    print("Time : {}".format(time.time() - time1))
    print(res)
    x = res.x
    A = [[0, 0, 0], [0, 0, 0]]
    payoff = [0, 0, 0]
    count = 0
    for i in range(3):
        for j in range(2):
            A[j][i] = res.x[count]
            count += 1
        payoff[i] = res.x[s+i]
    inter = res.nit
    err = res.fun
    print(A)
    print(payoff)
    print(inter)
    print(err)


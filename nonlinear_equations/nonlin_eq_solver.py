from math import inf

import numpy as np
import numpy.linalg.linalg



def jacobian(x1,x2,x3):
    return np.array([[2*x1,-8*x2,3*x3*x3],[4*x1,8*x2,-3],[ 2*x1,-2,2*x3]])

def function(x1, x2, x3):
    return np.array([x1 * x1 - 4 * x2 * x2 + x3 * x3 * x3 - 1,2 * x1 * x1 + 4 * x2 * x2 - 3 * x3,x1 * x1 - 2 * x2 + x3 * x3 - 1])


def newton1(w):
    epsilon = 0.001
    fun = np.array([inf, inf])
    i = 0
    while max(abs(fun)) > epsilon and not (i>100 and max(abs(fun)) > 0.5):
        jac = jacobian(w[0], w[1], w[2])
        fun = function(w[0], w[1], w[2])
        diff = np.linalg.solve(jac, -fun)
        w += diff
        i += 1
    return w, i

def newton2(w):
    epsilon = 0.001
    diff = np.array([inf, inf])
    i = 0
    while max(abs(diff)) > epsilon and not (i>100 and max(abs(diff)) > 0.5):
        jac = jacobian(w[0], w[1], w[2])
        fun = function(w[0], w[1], w[2])
        diff = np.linalg.solve(jac, -fun)
        w += diff
        i += 1
    return w, i



import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go


def explicit_euler_step_size_control(f, y0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = [0] * (N + 1)
    t_list = [0] * (N + 1)
    t_list[0] = t0
    h_list = []
    y[0] = y0
    eps_min = 10**(-2)
    eps_max = 10**(-4)
    h_min = h * 10**(-2)
    h_max = h * 10**2

    k = 0
    while t < t1:

        y_h = y[k] + h * f(t, y[k])
        y_h_half = y[k] + h/2 * f(t, y[k])
        error = abs(2*y_h_half - y_h)

        if error < eps_min and h < h_max:
            h *= 2

        elif error > eps_max and h > h_min:
            h /= 2

        else:
            y[k + 1] = y_h
            t = t + h
            t_list[k + 1] = t
            h_list.append(h)
            k += 1

    return np.array(y), t_list, h_list


def y_dash(t, y):
    return (1 - y/100)*y


y, t, h = explicit_euler_step_size_control(y_dash, 1, )

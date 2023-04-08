import numpy as np
from bokeh.plotting import figure, show
import plotly.graph_objects as go

t0 = 0
t1 = 10
h = 0.01


def euler_system(f, y0, t0, t1, h=0.01):
    """
    Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y = np.zeros((len(f), N + 1))
    y[:, 0] = y0
    for k in range(N):
        for i in range(len(f)):
            y[i, k + 1] = y[i, k] + h * f[i](t, y[:, k])
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def x_dash(t, x):
    return x[1]


def v_dash(t, x):
    return np.sin(t) - x[0]


y, t = euler_system([x_dash, v_dash], [0, 1], t0, t1, h)

p = figure(title="Oscillating Motion", x_axis_label='t')
p.line(t, y[0], legend_label="x", line_width=2, line_color="purple")
p.line(t, y[1], legend_label="v", line_width=2, line_color="blue")
show(p)

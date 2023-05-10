import numpy as np
from bokeh.plotting import figure, show
import plotly.graph_objects as go


def runge_kutta_4(f, y0, t0, t1, h):
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    t_list = [t]
    y = np.zeros((n_ode, nh))
    y[:, 0] = y0

    a21, a31, a32, a41, a42, a43 = 0.5, 0, 0.5, 0, 0, 1
    b1, b2, b3, b4 = 1/6, 1/3, 1/3, 1/6
    c1, c2, c3, c4 = 0, 0.5, 0.5, 1

    for i in range(nh):
        tau1, tau2, tau3, tau4 = t+c1*h, t+c2*h, t+c3*h, t+c4*h
        Y1 = y[:, i-1]
        Y2 = y[:, i-1] + h * (a21 * f(tau1, Y1))
        Y3 = y[:, i-1] + h * (a31 * f(tau1, Y1) + a32 * f(tau2, Y2))
        Y4 = y[:, i-1] + h * (a41 * f(tau1, Y1) + a42 * f(tau2, Y2) + a43 * f(tau3, Y3))
        y[:, i] = y[:, i-1] + h * (b1 * f(tau1, Y1) + b2 * f(tau2, Y2) + b3 * f(tau3, Y3) + b4 * f(tau4, Y4))
        t = t + h
        t_list.append(t)
    return y, t_list


def f(t, y):
    y0_dash = y[1] + np.sin(t)
    y1_dash = -y[0] + np.cos(t)
    return np.array([y0_dash, y1_dash])


y, t = runge_kutta_4(f, [2.5 * np.sin(2.5), -2.5 * np.cos(2.5)], -2.5, 10, 1/30)
print(y)

p = figure(title="Exercise 4.2: Runge Kutta of Order 4", x_axis_label='y1', y_axis_label='y2')
p.line(y[0, :], y[1, :], line_width=2, line_color="purple")
show(p)

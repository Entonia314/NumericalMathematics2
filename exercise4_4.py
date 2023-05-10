import numpy as np
from scipy.optimize import fsolve
from bokeh.plotting import figure, show
from bokeh.io import export_png
import plotly.graph_objects as go

s = 100
y0 = [2, 1]
t0, t1 = 0, 1
h = 0.01


def f(t, y):
    y1 = -s * y[0] + (s-1)*y[1]
    y2 = -y[1]
    return np.array([y1, y2])


def y_exact(t):
    return np.array([np.exp(-s*t) + np.exp(-t), np.exp(-t)])


def midpoint_residual(y_midpoint, f, t_old, y_old, t_midpoint):
    return y_midpoint - y_old - (t_midpoint - t_old) * f(t_midpoint, y_midpoint)


def implicit_midpoint(f, y0, t0, t1, h):
    n = int(np.ceil((t1 - t0) / h))
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)
    t = np.zeros(n + 1)
    y = np.zeros((m, n + 1))

    t[0] = t0
    y[:, 0] = y0

    for i in range(n):

        t_old = t[i]
        y_old = y[:, i]

        t_midpoint = t_old + 0.5*h
        y_midpoint = y_old + 0.5*h*f(t_old, y_old)
        y_midpoint = fsolve(midpoint_residual, y_midpoint, args=(f, t_old, y_old, t_midpoint))[0]

        t_new = t_old + h
        y_new = 2.0 * y_midpoint - y_old

        t[i+1] = t_new
        y[:, i+1] = y_new

    return y, t


def explicit_euler(f, y0, t0, t1, h):
    n = int(np.ceil((t1 - t0) / h))
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)
    t_list = np.zeros(n + 1)
    y = np.zeros((m, n + 1))
    t = t0

    t_list[0] = t0
    y[:, 0] = y0
    for k in range(n):
        y[:, k + 1] = y[:, k] + h * f(t, y[:, k])
        t = t + h
        t_list[k + 1] = t
    return y, t_list


for s in [10, 100, 1000]:
    y_mp, t_mp = implicit_midpoint(f, y0, t0, t1, h)
    y_e, t_e = explicit_euler(f, y0, t0, t1, h)
    y_real = y_exact(t_mp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_mp[0, :], y=y_mp[1, :], name="Implicit Midpoint"))
    fig.add_trace(go.Scatter(x=y_e[0, :], y=y_e[1, :], name="Explicit Euler"))
    fig.add_trace(go.Scatter(x=y_real[0, :], y=y_real[1, :], name="True solution"))
    fig.update_layout(title={'text': f"Exercise 4.4: Implicit Midpoint vs Explicit Euler with s={s}"},
                      xaxis_title='y1',
                      yaxis_title='y2',
                      template='simple_white'
                      )
    fig.write_image(f"exercise4-4_charts/exercise4_4_s{s}.png")

    """p = figure(title=f"Exercise 4.4: Implicit Midpoint vs Explicit Euler with s={s}", x_axis_label='y1', y_axis_label='y2')
    p.line(y_mp[0, :], y_mp[1, :], line_width=2, line_color="purple", legend_label="Implicit Midpoint")
    p.line(y_e[0, :], y_e[1, :], line_width=2, line_color="blue", legend_label="Explicit Euler")
    p.line(y_real[0, :], y_real[1, :], line_width=2, line_color="red", legend_label="True solution")
    export_png(p, filename=f"exercise4-4_charts/exercise4_4_s{s}.png")"""



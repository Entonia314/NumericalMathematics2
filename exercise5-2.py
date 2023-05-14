import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go


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


def bdf_equation(y_n, f, y_n1, y_n2, t_n, h):
    return 3 * y_n - 4 * y_n1 + y_n2 - 2 * h * f(t_n, y_n)


def bdf_2step(f, y0, t0, t1, h):
    n = int(np.ceil((t1 - t0) / h))
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)
    t = np.linspace(t0, t1, num=int(np.ceil((t1 - t0) / h)))
    y = np.zeros((m, n + 1))
    e = np.zeros(n + 1)

    y[:, 0:2] = np.array(explicit_euler(f, y0, t0, t0 + h, h)[0])

    for i in range(2, n):
        t_i = t[i]
        y_i1 = y[:, i - 1]
        y_i2 = y[:, i - 2]
        y_i = y_i1 + h * f(t_i, y_i1)

        y_i = fsolve(bdf_equation, y_i, args=(f, y_i1, y_i2, t_i, h))

        y[:, i] = y_i
        e[i] = euclidean(y_i, y_i1)

    return y, t, e


def y_dash(t, y):
    return np.array([-1000 * y[0] + 999 * y[1], -y[1]])


def f(t, y):
    y0_dash = y[1] + np.sin(t)
    y1_dash = -y[0] + np.cos(t)
    return np.array([y0_dash, y1_dash])


fig = go.Figure()
fig_e = go.Figure()
fig_euler = go.Figure()

for h in [0.1, 0.05, 0.01, 0.001]:
    y_bdf, t_bdf, e_bdf = bdf_2step(y_dash, [2, 1], 0, 1, h)

    fig.add_trace(go.Scatter(x=y_bdf[0, :], y=y_bdf[1, :], name=f"h={h}"))
    fig_e.add_trace(go.Scatter(x=t_bdf, y=e_bdf, name=f"h={h}"))

y_ee, t_ee = explicit_euler(y_dash, [2, 1], 0, 1, 0.1)
fig_euler.add_trace(go.Scatter(x=y_ee[:, 0], y=y_ee[:, 1], name=f"h={0.001}"))

fig.update_layout(title={'text': f"Exercise 5.2: 2-step BDF - Plotting y_1 against y_2"},
                  xaxis_title='y_1',
                  yaxis_title='y_2',
                  template='simple_white'
                  )
fig_e.update_layout(title={'text': f"Exercise 5.2: 2-step BDF - Convergence Rate"},
                    xaxis_title='t',
                    yaxis_title='Distance between y_{i} and y_{i-1}',
                    template='simple_white'
                    )
fig_euler.update_layout(title={'text': f"Exercise 5.2: Explicit Euler"},
                        xaxis_title='y_1',
                        yaxis_title='y_2',
                        template='simple_white'
                        )
fig.write_image(f"exercise5-2_charts/exercise5_2_y.png")
fig_e.write_image(f"exercise5-2_charts/exercise5_2_e.png")
fig_euler.write_image(f"exercise5-2_charts/exercise5_2_euler.png")

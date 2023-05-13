import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go


def explicit_euler(f, y0, t0, t1, h):
    """
    Explicit Euler method for a differential equation: y' = f(t, y).
    :param f: function
    :param y0: float or int, initial value y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = [0] * (N + 1)
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[0] = y0
    for k in range(N):
        y[k + 1] = y[k] + h * f(t, y[k])
        t = t + h
        t_list[k + 1] = t
    return np.array(y), t_list


def runge_kutta_4(f, y0, t0, t1, h):
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    t_list = [t]
    y = np.zeros((n_ode, nh))
    y[:, 0] = y0

    a21, a31, a32, a41, a42, a43 = 0.5, 0, 0.5, 0, 0, 1
    b1, b2, b3, b4 = 1 / 6, 1 / 3, 1 / 3, 1 / 6
    c1, c2, c3, c4 = 0, 0.5, 0.5, 1

    for i in range(nh):
        tau1, tau2, tau3, tau4 = t + c1 * h, t + c2 * h, t + c3 * h, t + c4 * h
        Y1 = y[:, i - 1]
        Y2 = y[:, i - 1] + h * (a21 * f(tau1, Y1))
        Y3 = y[:, i - 1] + h * (a31 * f(tau1, Y1) + a32 * f(tau2, Y2))
        Y4 = y[:, i - 1] + h * (a41 * f(tau1, Y1) + a42 * f(tau2, Y2) + a43 * f(tau3, Y3))
        y[:, i] = y[:, i - 1] + h * (b1 * f(tau1, Y1) + b2 * f(tau2, Y2) + b3 * f(tau3, Y3) + b4 * f(tau4, Y4))
        t = t + h
        t_list.append(t)
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

    y[:, 0:2] = np.array(explicit_euler(f, y0, t0, t0 + h, h)[0]).transpose()

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

for h in [0.1, 0.05, 0.01, 0.001]:
    y_bdf, t_bdf, e_bdf = bdf_2step(y_dash, [2, 1], 0, 10, h)

    fig.add_trace(go.Scatter(x=y_bdf[0, :], y=y_bdf[1, :], name=f"h={h}"))
    fig_e.add_trace(go.Scatter(x=t_bdf, y=e_bdf, name=f"h={h}"))

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
fig.write_image(f"exercise5-2_charts/exercise5_2_y.png")
fig_e.write_image(f"exercise5-2_charts/exercise5_2_e.png")

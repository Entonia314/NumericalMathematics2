import numpy as np
from bokeh.plotting import figure, show
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
    y = [0]*(N+1)
    t_list = [0]*(N+1)
    t_list[0] = t0
    y[0] = y0
    for k in range(N):
        y[k + 1] = y[k] + h * f(t, y[k])
        t = t + h
        t_list[k+1] = t
    return y, t_list


def y_dash(t, y):
    return ( 1 - y * 1/100 ) * y


def y(t):
    return 1 / ( 1/100 + 99/100 * 1/np.exp(t))


fig = go.Figure()
p = figure(title="Exercise 4.1: Explicit Euler", x_axis_label='t', y_axis_label='y')
t = np.linspace(0, 10, 100)
y_real = y(t)
fig.add_trace(go.Scatter(x=t, y=y_real, name=f"Analytical y"))
p.line(t, y_real, line_width=2, line_color="red", legend_label="Analytical y")

for h in [1, 0.5, 0.1, 0.05, 0.01]:
    y_euler, t_list = explicit_euler(y_dash, 1, 0, 10, h)
    fig.add_trace(go.Scatter(x=t_list, y=y_euler, name=f"Explicit Euler with h={h}"))
    p.line(t_list, y_euler, line_width=2, legend_label=f"Explicit Euler with h={h}")

fig.update_layout(title={'text': f"Exercise 4.1: Explicit Euler Method"},
                  xaxis_title='t',
                  yaxis_title='y',
                  )
#fig.show()
show(p)

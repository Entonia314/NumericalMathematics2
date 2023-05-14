import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def explicit_euler_step_size_control(f, y0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = [y0]
    t_list = [t0]
    h_list = [h]
    error_list = []
    eps_min = 0.001
    eps_max = 0.1
    h_min = 0.0001
    h_max = 1
    f_min = 0.5
    f_max = 2
    max_loops = 5

    k = 0
    loop_control_count = 0
    while t < t1:

        y_h = y[k] + h * f(t, y[k])
        y_halved_h = y[k] + h/2 * f(t, y[k])
        error = abs(y_h - y_halved_h)

        if error < eps_min and h < h_max and loop_control_count < max_loops:
            h *= f_max
            print(f"h={h}, error={error}, t={t}: h will be doubled")
            loop_control_count += 1

        elif error > eps_max and h > h_min and loop_control_count < max_loops:
            h *= f_min
            print(f"h={h}, error={error}, t={t}: will be halved")
            loop_control_count += 1

        else:
            y.append(y_h)
            t = t + h
            t_list.append(t)
            h_list.append(h)
            k += 1
            error_list.append(error)
            loop_control_count = 0

    return np.array(y), t_list, h_list, error_list


def y_dash(t, y):
    return (1 - y/100)*y


h_start = 0.1
fig = make_subplots(specs=[[{"secondary_y": True}]])
y, t, h, e = explicit_euler_step_size_control(y_dash, 1, 0, 10, h_start)

fig.add_trace(go.Scatter(x=t, y=y, name=f"y"), secondary_y=False)
fig.add_trace(go.Scatter(x=t, y=h, name=f"h"), secondary_y=True)

fig.update_layout(title={'text': f"Exercise 5.3: Explicit Euler with step size control"},
                  xaxis_title='t',
                  yaxis_title='',
                  template='simple_white'
                  )

fig.write_image(f"exercise5-3_charts/exercise5_3.png")

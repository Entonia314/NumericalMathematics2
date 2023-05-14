import numpy as np
from numpy.random import uniform
import pandas as pd
from numpy import expm1
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mc_integration(x_interval, num_mc):
    dimension = x_interval.shape[0]
    sum_m = []
    for i in range(num_mc):
        product = 1
        for j in range(1, dimension):
            x = uniform(x_interval[j, 0], x_interval[j, 1])
            product *= np.exp(x/j)
        sum_m.append(product)
    result = sum(sum_m)/num_mc
    dev = np.std(sum_m)
    return result, dev


def integral_real(x_interval):
    dimension = x_interval.shape[0]
    product = 1
    for j in range(1, dimension):
        product *= j * (np.exp(1/j)-1)
    return product


dimensions = [1, 2, 3, 5, 10, 20, 30, 40, 50]
results_mc = []
error = []
deviation = []
real_solution = []

df = pd.DataFrame(index=dimensions)

for d in dimensions:

    x_d = np.array([[0, 1]*d])
    x_d = x_d.reshape((d, 2))

    y, std = mc_integration(x_d, 10000)

    results_mc.append(y)
    deviation.append(std)
    real_solution.append(integral_real(x_d))
    error.append(abs(integral_real(x_d)-y))

df["MC Integration"] = results_mc
df["Error"] = error
df["Deviation"] = deviation

print(results_mc)
print(error)
print(deviation)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=dimensions, y=error, name=f"Error"), secondary_y=True)
# fig.add_trace(go.Scatter(x=dimensions, y=real_solution, name=f"Analytical Solution"), secondary_y=False)
fig.add_trace(go.Scatter(x=dimensions, y=results_mc, name=f"Integral via MC", error_y=dict(type="data", array=deviation, visible=True)), secondary_y=False)

fig.update_layout(title={'text': f"Exercise 5.4: Integration via Monte Carlo"},
                  xaxis_title='Dimension',
                  yaxis_title='',
                  template='simple_white'
                  )

fig.write_image(f"exercise5-4_charts/exercise5_4.png")

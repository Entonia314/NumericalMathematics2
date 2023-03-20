import numpy as np
from scipy.sparse import diags, identity
import matplotlib.pyplot as plot
import plotly.express as px
import plotly.graph_objects as go
import pandas


def steepest_descent(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    x = x0
    r = vector_b - matrix_a @ x
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        p = matrix_a @ r
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ p)
        x = x + alpha * r
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, ((cond_number - 1) / (cond_number + 1)) ** k * e0)
        r = r - alpha * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def conjugate_gradient(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    x = x0
    r = vector_b - matrix_a @ x
    p = r
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        a_p = matrix_a @ p
        alpha = (np.transpose(r) @ r) / (np.transpose(p) @ a_p)
        x = x + alpha * p
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, 2 * ((np.sqrt(cond_number) - 1) / (np.sqrt(cond_number) + 1)) ** k * e0)
        r_next = r - alpha * a_p
        beta = (np.transpose(r_next) @ r_next) / (np.transpose(r) @ r)
        r = r_next
        p = r + beta * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def draw_plot(max_k=10000, eps=1e-10, dim=10):
    K1d = diags([-1, 2, -1], [-1, 0, 1], shape=(dim, dim)).toarray()
    id_n = identity(dim).toarray()
    A = np.kron(id_n, K1d) + np.kron(K1d, id_n)
    b = np.random.randint(0, 10, dim ** 2)
    x0 = np.zeros(dim ** 2)

    x_sd, k_sd, e_sd, e_bound_sd = steepest_descent(A, b, x0, max_k=max_k, eps=eps)
    x_cg, k_cg, e_cg, e_bound_cg = conjugate_gradient(A, b, x0, max_k=max_k, eps=eps)

    eps_array = np.repeat(eps, max(k_sd, k_cg))

    if k_sd > k_cg:
        fill_array = np.zeros(abs(k_sd - k_cg))
        e_cg = np.append(e_cg, fill_array)
    else:
        fill_array = np.zeros(abs(k_sd - k_cg))
        e_sd = np.append(e_sd, fill_array)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_sd, name='SD real', marker={'color': 'red'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_bound_sd, name='SD theoretical bound', marker={'color': 'orange'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_cg, name='CG real', marker={'color': 'blue'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_bound_cg, name='CG theoretical bound', marker={'color': 'cyan'}))
    fig.update_yaxes(type="log")
    fig.update_layout(title={'text': 'Relative Energy Error of Steepest Descent and Conjugate Gradient',
                             'font': dict(size=18)},
                      xaxis_title='Iterations',
                      yaxis_title='Relative Energy Error',
                      legend=dict(y=0.5, font_size=16),
                      showlegend=True
                      )

    fig.write_image(str("relEnergyError_N" + str(dim) + ".png"))


for n in [10, 100, 10000]:
    draw_plot(dim=n)



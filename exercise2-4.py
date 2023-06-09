import numpy as np
from scipy.sparse import diags, identity, csr_matrix, dia_matrix, csr_array
import plotly.graph_objects as go
import time

start_program = time.time()


def steepest_descent(matrix_a, vector_b, x0, max_k=1000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    matrix_a = dia_matrix(matrix_a)

    x = x0
    vector_b = vector_b
    r = vector_b - matrix_a @ x
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        if k == 0:
            start_sd_it = time.time()
        p = matrix_a @ r    # Matrix multiplication
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ p)
        x = x + alpha * r
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, ((cond_number - 1) / (cond_number + 1)) ** k * e0)
        r = r - alpha * p
        result_x = np.append(result_x, x, axis=0)

        if k == 0:
            end_sd_it = time.time()
            print("One iteration of SD with n = ", n, " needs ", end_sd_it - start_sd_it, " seconds.")

        k += 1

    print("SD done, k = ", k)
    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def conjugate_gradient(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    matrix_a = dia_matrix(matrix_a)

    x = x0
    r = vector_b - (matrix_a @ x)
    p = r
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        if k == 0:
            start_cg_it = time.time()
        a_p = (matrix_a @ p)    # only matrix multiplication
        alpha = (r.transpose() @ r) / (p.transpose() @ a_p)
        x = x + alpha * p
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, 2 * ((np.sqrt(cond_number) - 1) / (np.sqrt(cond_number) + 1)) ** k * e0)
        r_next = r - alpha * a_p
        beta = (r_next.transpose() @ r_next) / (r.transpose() @ r)
        r = r_next
        p = r + beta * p
        result_x = np.append(result_x, x, axis=0)

        if k == 0:
            end_cg_it = time.time()
            print("One iteration of CG with n = ", n, " needs ", end_cg_it - start_cg_it, " seconds.")

        k += 1
    print("CG done, k = ", k)
    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def conjugate_gradient_np(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    row_ind = np.array(range(0, n**2))
    col_ind = np.zeros(n**2, dtype=int)

    matrix_a = csr_matrix(matrix_a)
    matrix_b = csr_matrix((vector_b, (row_ind, col_ind)), shape=(n**2, n**2))
    matrix_b[:, 0].toarray().flatten()

    x = x0
    matrix_x = csr_matrix((x, (row_ind, col_ind)), shape=(n**2, n**2))
    r = vector_b - matrix_a @ x
    p = r
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        if k == 0:
            start_cg_it = time.time()
        a_p = matrix_a @ p
        alpha = (r.transpose() @ r) / (p.transpose() @ a_p)
        x = x + alpha * p
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, 2 * ((np.sqrt(cond_number) - 1) / (np.sqrt(cond_number) + 1)) ** k * e0)
        r_next = r - alpha * a_p
        beta = (r_next.transpose() @ r_next) / (r.transpose() @ r)
        r = r_next
        p = r + beta * p
        result_x = np.append(result_x, x, axis=0)
        if k == 0:
            end_cg_it = time.time()
            print("One iteration of CG without sparse format with n = ", n, " needs ", end_cg_it - start_cg_it, " seconds.")

        k += 1
    print("CG done, k = ", k)
    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def draw_plot(max_k=10000, eps=1e-10, dim=10):

    print("\nn = ", dim, " and therefore N = nxn = ", dim*dim)
    a_start = time.time()
    K1d = diags([-1, 2, -1], [-1, 0, 1], shape=(dim, dim)).toarray()
    id_n = identity(dim).toarray()
    A = np.kron(id_n, K1d) + np.kron(K1d, id_n)
    a_end = time.time()
    print("Calculating A needs ", a_end - a_start, " seconds.")

    rand_start = time.time()
    b = np.random.randint(0, 10, dim ** 2)
    rand_end = time.time()
    print("Generating a randomized b needs ", rand_end - rand_start, " seconds.")

    x0 = np.ones(dim ** 2)

    print("\n---Steepest Descent-----------------------------------------")
    start_sd = time.time()
    x_sd, k_sd, e_sd, e_bound_sd = steepest_descent(A, b, x0, max_k=1000, eps=eps)
    end_sd = time.time()
    print("Stochastic descent for n = ", dim, " and therefore N = nxn = ", dim*dim, " needed ", end_sd - start_sd, "seconds.")

    print("\n---Gradient Conjugate---------------------------------------")
    start_cg = time.time()
    x_cg, k_cg, e_cg, e_bound_cg = conjugate_gradient(A, b, x0, max_k=max_k, eps=eps)
    end_cg = time.time()
    print("Conjugate gradient for n = ", dim, " and therefore N = nxn = ", dim*dim, " needed ", end_cg - start_cg, "seconds.")

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

    fig.write_image(str("exercise2-4_charts/relEnergyError_N" + str(dim) + ".png"))


for n in [20]:
    draw_plot(dim=n)


end_program = time.time()
print("\n--------------------------------------------------------------")
print("Whole program needed ", end_program - start_program, "seconds.")

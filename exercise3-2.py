import numpy as np
from scipy.sparse import diags, identity, csc_matrix, dia_matrix, csr_array
from scipy.sparse.linalg import splu, spilu, LinearOperator, inv
import plotly.graph_objects as go
import time
import sys

start_program = time.time()


def sparse_cholesky(A):  # The input matrix A must be a sparse symmetric positive-definite.

    n = A.shape[0]
    LU = splu(A, diag_pivot_thresh=0)  # sparse LU decomposition

    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():  # check the matrix A is positive definite.
        return LU.L.dot(diags(LU.U.diagonal() ** 0.5))
    else:
        sys.exit('The matrix is not positive definite')


def conjugate_gradient(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    matrix_a = dia_matrix(matrix_a)     # sparse diagonal matrix

    x = x0
    r = vector_b - (matrix_a @ x)
    p = r
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        if k == 0:
            start_cg_it = time.time()
        a_p = (matrix_a @ p)
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


def conjugate_gradient_preconditioning(matrix_a, vector_b, x0, matrix_m, max_k=10000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    matrix_a = dia_matrix(matrix_a)     # sparse diagonal matrix

    x = x0
    r = vector_b - (matrix_a @ x)
    r_hat = matrix_m.matvec(r)
    p = r_hat
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        if k == 0:
            start_cg_it = time.time()
        a_p = (matrix_a @ p)
        alpha = (r.transpose() @ r_hat) / (p.transpose() @ a_p)
        x = x + alpha * p
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, 2 * ((np.sqrt(cond_number) - 1) / (np.sqrt(cond_number) + 1)) ** k * e0)
        r_next = r - alpha * a_p
        r_hat_next = matrix_m.matvec(r_next)
        beta = (r_next.transpose() @ r_hat_next) / (r.transpose() @ r_hat)
        r = r_next
        r_hat = r_hat_next
        p = r_hat + beta * p
        result_x = np.append(result_x, x, axis=0)

        if k == 0:
            end_cg_it = time.time()
            print("One iteration of CG with preconditioning with n = ", n, " needs ", end_cg_it - start_cg_it, " seconds.")

        k += 1
    print("CGp done, k = ", k)
    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


def draw_plot(max_k=10000, eps=1e-10, dim=10):

    print("\nn = ", dim, " and therefore N = nxn = ", dim*dim)
    a_start = time.time()
    K1d = diags([-1, 2, -1], [-1, 0, 1], shape=(dim, dim)).toarray()
    id_n = identity(dim).toarray()
    A = np.kron(id_n, K1d) + np.kron(K1d, id_n)
    a_end = time.time()
    print("Calculating A needs ", a_end - a_start, " seconds.")
    print("A: \n", A, '\n')

    m_start = time.time()
    M2 = splu(csc_matrix(A))
    M_x = lambda x: M2.solve(x)
    Minv = LinearOperator(csc_matrix(A).shape, M_x)
    m_end = time.time()
    print(f"Calculating M needs {m_end-m_start} seconds.")

    rand_start = time.time()
    b = np.random.randint(0, 10, dim ** 2)
    rand_end = time.time()
    print("Generating a randomized b needs ", rand_end - rand_start, " seconds.")

    x0 = np.ones(dim ** 2)

    print("\n---Gradient Conjugate without preconditioning-----------------------------------------")
    start_sd = time.time()
    x_sd, k_sd, e_sd, e_bound_sd = conjugate_gradient(A, b, x0, max_k=1000, eps=eps)
    end_sd = time.time()
    print("Gradient Conjugate without preconditioning for n = ", dim, " and therefore N = nxn = ", dim*dim, " needed ", end_sd - start_sd, "seconds.")

    print("\n---Gradient Conjugate with preconditioning---------------------------------------")
    start_cg = time.time()
    x_cg, k_cg, e_cg, e_bound_cg = conjugate_gradient_preconditioning(A, b, x0, matrix_m=Minv, max_k=max_k, eps=eps)
    end_cg = time.time()
    print("Conjugate gradient with preconditioning for n = ", dim, " and therefore N = nxn = ", dim*dim, " needed ", end_cg - start_cg, "seconds.")

    if k_sd > k_cg:
        fill_array = np.zeros(abs(k_sd - k_cg))
        e_cg = np.append(e_cg, fill_array)
    else:
        fill_array = np.zeros(abs(k_sd - k_cg))
        e_sd = np.append(e_sd, fill_array)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_sd, name='CG real', marker={'color': 'red'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_bound_sd, name='CG theoretical bound', marker={'color': 'orange'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_cg, name='CGp real', marker={'color': 'blue'}))
    fig.add_trace(go.Scatter(x=np.array(range(max(k_sd, k_cg))), y=e_bound_cg, name='CGp theoretical bound', marker={'color': 'cyan'}))
    fig.update_yaxes(type="log")
    fig.update_layout(title={'text': f'Relative Energy Error of Conjugate Gradient for n={dim}, N={dim**2}',
                             'font': dict(size=18)},
                      xaxis_title='Iterations',
                      yaxis_title='Relative Energy Error',
                      legend=dict(y=0.5, font_size=16),
                      showlegend=True
                      )

    fig.write_image(str("exercise3-2_charts/relEnergyError_N" + str(dim**2) + ".png"))


for n in [40]:
    draw_plot(dim=n)


end_program = time.time()
print("\n--------------------------------------------------------------")
print("Whole program needed ", end_program - start_program, "seconds.")

import numpy as np
from scipy.sparse import diags, identity, csr_matrix, dia_matrix
import plotly.graph_objects as go
import time

start_program = time.time()


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


def draw_plot(max_k=10000, eps=1e-20, dim=10, m=5, eps_eigenvalue=1e-4):

    large_eigenvalues = np.random.uniform(90, 110, m)
    clustered_eigenvalues = np.random.uniform(1-eps_eigenvalue, 1+eps_eigenvalue, dim-m)
    eigenvalues = np.concatenate((clustered_eigenvalues, large_eigenvalues))
    eigenvalues.sort()
    #print("Eigenvalues of A: ", eigenvalues)
    A = np.diag(eigenvalues)

    b = np.random.randint(-110, -100, dim)
    x0 = np.random.randint(100, 110, dim)

    start_cg = time.time()
    x_cg, k_cg, e_cg, e_bound_cg = conjugate_gradient(A, b, x0, max_k=max_k, eps=eps)
    end_cg = time.time()
    print("Conjugate gradient for n = ", dim, " needed ", end_cg - start_cg, "seconds.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(k_cg)), y=e_cg, name='CG real', marker={'color': 'blue'}))
    fig.add_trace(go.Scatter(x=np.array(range(k_cg)), y=e_bound_cg, name='CG theoretical bound', marker={'color': 'cyan'}))
    fig.add_vline(x=m)
    fig.update_yaxes(type="log")
    fig.update_layout(title={'text': str('Relative Energy Error of Conjugate Gradient, n='+str(dim)+', m='+str(m)),
                             'font': dict(size=18)},
                      xaxis_title='Iterations',
                      yaxis_title='Relative Energy Error',
                      legend=dict(y=0.5, font_size=16),
                      showlegend=True
                      )

    fig.write_image(str("exercise2-5_charts/relEnergyError_n" + str(n) + "_m"+str(m)+".png"))


for n in [1000]:
    draw_plot(dim=n, m=10)


end_program = time.time()
print("\n--------------------------------------------------------------")
print("Whole program needed ", end_program - start_program, "seconds.")

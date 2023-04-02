from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
import numpy as np
import plotly.graph_objects as go
import time


def draw_plot(max_k=10000, eps=1e-5, dim=10, m=5, eps_eigenvalue=1e-4, restart=20):

    large_eigenvalues = np.random.uniform(90, 110, m)
    clustered_eigenvalues = np.random.uniform(1-eps_eigenvalue, 1+eps_eigenvalue, dim-m)
    eigenvalues = np.concatenate((clustered_eigenvalues, large_eigenvalues))
    eigenvalues.sort()
    print("Eigenvalues of A: ", eigenvalues)
    A = np.diag(eigenvalues)

    b = np.random.randint(-110, -100, dim)
    x0 = np.random.randint(100, 110, dim)

    start_1 = time.time()
    x1, info1 = gmres(A, b, x0, eps, restart, max_k)
    end_1 = time.time()
    print("GMRES for A = \n", A, " without precoinditioning needed ", end_1 - start_1, "seconds.")
    if info1 == 0:
        print("Convergence tolerance achieved.")
    else:
        print("Convergence tolerance not achieved, ended after ", info1, " iterations.")
    print("Solution x: \n", x1)

    """fig = go.Figure()
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

    fig.write_image(str("exercise2-5_charts/relEnergyError_n" + str(n) + "_m"+str(m)+".png"))"""


for n in [10]:
    draw_plot(dim=n, m=10)

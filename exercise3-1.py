from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres, spilu, spsolve, LinearOperator
from scipy.sparse import csr_matrix, csc_matrix
from numpy.linalg import cond
from scipy.io import mmread
import pyamg
import numpy as np
import plotly.graph_objects as go
import time


def do_gmres(A, b, x0, preconditioning=True, restart=20):

    condition_number = cond(A.toarray())

    if preconditioning:
        M2 = spilu(A)
        M_x = lambda x: M2.solve(x)
        M = LinearOperator(np.shape(A), M_x)
        start_1 = time.time()
        residuals = []
        result, info1 = pyamg.krylov.gmres(A=A, b=b, x0=x0, M=M, restrt=restart, maxiter=100, tol=0.0001, residuals=residuals)
    else:
        start_1 = time.time()
        residuals = []
        result, info1 = pyamg.krylov.gmres(A=A, b=b, x0=x0, M=None, restrt=restart, maxiter=100, tol=0.0001, residuals=residuals)

    end_1 = time.time()

    if preconditioning:
        print(f"GMRES for A with dimensions {np.shape(A.toarray())} and condition number {round(condition_number)} with preconditioning and restart parameter m={restart} needed ", round(end_1 - start_1, 2), "seconds.")
        if info1 == 0:
            print("Convergence tolerance achieved.\n")
        else:
            print("Convergence tolerance not achieved, ended after ", info1, " iterations.\n")
        # print("Solution x: \n", result)
    else:
        print(f"GMRES for A with dimensions {np.shape(A.toarray())} and condition number {round(condition_number)} without preconditioning and restart parameter m={restart} needed ", round(end_1 - start_1, 2), "seconds.")
        if info1 == 0:
            print("Convergence tolerance achieved.\n")
        else:
            print("Convergence tolerance not achieved, ended after ", info1, " iterations.\n")
        # print("Solution x: \n", result)
    return residuals


# First Matrix

print("------Matrix #1: BP 200: Original Harwell sparse matrix test collection Simplex method basis matrix----------\n")
print("------without preconditioning----------\n")

A1 = mmread("matrices/bp___200.mtx.gz")
A1 = csc_matrix(A1)
b1 = np.ones((np.shape(A1)[0], 1))
x1 = np.zeros((np.shape(A1)[0], 1))

fig = go.Figure()
figp = go.Figure()

for m in [10, 20, 50, 100]:
    res1 = do_gmres(A1, b1, x1, preconditioning=False, restart=m)
    fig.add_trace(
        go.Scatter(x=np.array(range(len(res1))), y=res1, name=f"gmres m={m}"))

print("------with preconditioning----------\n")

for m in [10, 20, 50, 100]:
    res1p = do_gmres(A1, b1, x1, preconditioning=True, restart=m)
    fig.add_trace(
        go.Scatter(x=np.array(range(len(res1p))), y=res1p, name=f"pgmres m={m}"))
    figp.add_trace(
        go.Scatter(x=np.array(range(len(res1p))), y=res1p, name=f"pgmres m={m}"))


fig.update_yaxes(type="log")
fig.update_layout(title={'text': f"Residuals of GMRES: Matrix 1",
                         'font': dict(size=18)},
                  xaxis_title='Iterations',
                  yaxis_title='Residuals',
                  legend=dict(y=0.5, font_size=16),
                  showlegend=True
                  )
fig.write_image(f"exercise3-1_charts/matrix1.png")

figp.update_yaxes(type="log")
figp.update_layout(title={'text': f"Residuals of GMRES: Matrix 1",
                          'font': dict(size=18)},
                   xaxis_title='Iterations',
                   yaxis_title='Residuals',
                   legend=dict(y=0.5, font_size=16),
                   showlegend=True
                   )
figp.write_image(f"exercise3-1_charts/matrix1_p.png")

# Second Matrix

print("------Matrix #2: FS 183 1: Chemical kinetics problems - PSMOG atmospheric polution study -- 1st output time step----------\n")
A2 = mmread("matrices/fs_183_1.mtx.gz")
A2 = csc_matrix(A2)
b2 = np.ones((np.shape(A2)[0], 1))
x2 = np.zeros((np.shape(A2)[0], 1))

fig = go.Figure()

print("------without preconditioning----------\n")

for m in [10, 50, 100]:
    res2 = do_gmres(A2, b2, x2, preconditioning=False, restart=m)
    fig.add_trace(
        go.Scatter(x=np.array(range(len(res2))), y=res2, name=f"gmres m={m}"))

print("------with preconditioning----------\n")

for m in [10, 50, 100]:
    res2p = do_gmres(A2, b2, x2, preconditioning=True, restart=m)
    fig.add_trace(
        go.Scatter(x=np.array(range(len(res2p))), y=res2p, name=f"pgmres m={m}"))

fig.update_yaxes(type="log")
fig.update_layout(title={'text': f"Residuals of GMRES: Matrix 2",
                         'font': dict(size=18)},
                  xaxis_title='Iterations',
                  yaxis_title='Residuals',
                  legend=dict(y=0.5, font_size=16),
                  showlegend=True
                  )

fig.write_image(f"exercise3-1_charts/matrix2.png")

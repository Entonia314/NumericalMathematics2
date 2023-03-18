import numpy as np

A = np.array([[4, 0.5, 1], [2, 0.5, 1], [4, 1, 0.9]])
b = np.array([1, 1, 1])


def damped_richardson(A, b, eps, x0, omega=1):
    print("Damped Richardson Method for: \n", A, b)
    num_iterations = 0
    error_too_big = True
    res = 0

    while error_too_big and num_iterations < 100:
        x1 = x0 + omega * (b - A @ x0.transpose())
        if np.linalg.norm(x0 - x1) < eps:
            error_too_big = False
        res = np.linalg.norm(x0 - x1)
        x0 = x1
        num_iterations += 1

    print("Iterations: ", num_iterations)
    print("Residual error: ", res)
    print("x*: ", x1)

    return x1


def jacobi(A, b, eps, x0, omega=1):
    print("Jacobi Method for: \n", A, b)
    D = np.tril(np.triu(A))
    D_inv = np.linalg.inv(D)
    num_iterations = 0
    error_too_big = True
    res = 0

    while error_too_big and num_iterations < 100:
        x1 = x0 + omega * D_inv @ (b - A @ x0.transpose()).transpose()
        if np.linalg.norm(x0 - x1) < eps:
            error_too_big = False
        res = np.linalg.norm(x0 - x1)
        x0 = x1
        num_iterations += 1

    print("Iterations: ", num_iterations)
    print("Residual error: ", res)
    print("x*: ", x1)

    return x1


def gauss_seidel(A, b, eps, x0):
    print("Gauss Seidel Method for: \n", A, b)
    U = np.triu(A, 1)
    L = np.tril(A, -1)
    D = np.tril(np.triu(A))
    D_L_inv = np.linalg.inv(D + L)
    num_iterations = 0
    error_too_big = True
    res = 0

    while error_too_big and num_iterations < 100:
        x1 = D_L_inv @ (b - U @ x0.transpose()).transpose()
        if np.linalg.norm(x0 - x1) < eps:
            error_too_big = False
        res = np.linalg.norm(x0 - x1)
        x0 = x1
        num_iterations += 1

    print("Iterations: ", num_iterations)
    print("Residual error: ", res)
    print("x*: ", x1)

    return x1


damped_richardson(A=A, b=b, eps=0.00001, x0=np.array([0, 0, 0]), omega=1)
jacobi(A=A, b=b, eps=0.00001, x0=np.array([0, 0, 0]), omega=1)
gauss_seidel(A=A, b=b, eps=0.00001, x0=np.array([0, 0, 0]))


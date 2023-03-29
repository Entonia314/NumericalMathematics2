import numpy as np
from scipy.sparse import csr_matrix

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([14, 32, 50])


def steepest_descent(matrix_a, vector_b, x0, max_k=1000, eps=1e-10):
    cond_number = np.linalg.cond(matrix_a)
    k = 0
    result = np.array([])
    e_array = np.array([])
    error_bound = np.array([])

    matrix_a = csr_matrix(matrix_a)     # Compressed sparse row matrix

    x = x0
    r = vector_b - matrix_a @ x
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        p = matrix_a @ r    # only matrix multiplication
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ p)
        x = x + alpha * r
        e_array = np.append(e_array, [np.linalg.norm(r) / e0], axis=0)
        error_bound = np.append(error_bound, ((cond_number - 1) / (cond_number + 1)) ** k * e0)
        r = r - alpha * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k + 1, len(vector_b))), k, e_array, error_bound


x, iterations, error_real, error_theoretical = steepest_descent(A, b, np.array([0, 0, 0]))
print("x: \n", x)
print("Number of iterations: ", iterations)

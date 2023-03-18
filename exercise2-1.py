import numpy as np

A = np.array([[4, 0.5, 1], [0.5, 2, 1], [1, 1, 1]])
b = np.array([1, 4, 6])


def steepest_descent(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    k = 0
    result = np.array([])

    x = x0
    r = vector_b - matrix_a @ x

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        p = A @ r
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ p)
        x = x + alpha * r
        r = r - alpha * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k+1, len(vector_b))), k


x, iterations = steepest_descent(A, b, np.array([0, 0, 0]))
print("Result of x: ", x)
print("Number of iterations: ", iterations)

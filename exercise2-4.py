import numpy as np
from scipy.sparse import diags, identity


def steepest_descent(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    k = 0
    result = np.array([])
    e_array = np.array([])

    x = x0
    r = vector_b - matrix_a @ x
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        p = A @ r
        alpha = (np.transpose(r) @ r) / (np.transpose(r) @ p)
        x = x + alpha * r
        e_array = np.append(e_array, [np.linalg.norm(r)/e0], axis=0)
        r = r - alpha * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k+1, len(vector_b))), k, e_array


def conjugate_gradient(matrix_a, vector_b, x0, max_k=10000, eps=1e-10):
    k = 0
    result = np.array([])
    e_array = np.array([])

    x = x0
    r = vector_b - matrix_a @ x
    p = r
    e0 = np.linalg.norm(r)

    result_x = np.append(result, x, axis=0)

    while k < max_k and any(abs(r) > eps):
        a_p = matrix_a @ p
        alpha = (np.transpose(r) @ r) / (np.transpose(p) @ a_p)
        x = x + alpha * p
        e_array = np.append(e_array, [np.linalg.norm(r)/e0], axis=0)
        r_next = r - alpha * a_p
        beta = (np.transpose(r_next) @ r_next) / (np.transpose(r) @ r)
        r = r_next
        p = r + beta * p
        result_x = np.append(result_x, x, axis=0)

        k += 1

    return result_x.reshape((k+1, len(vector_b))), k, e_array


n = 5

K1d = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
id_n = identity(n).toarray()
A = np.kron(id_n, K1d) + np.kron(K1d, id_n)

b = np.random.randint(0, 10, n**2)

x_sd, k_sd, e_sd = steepest_descent(A, b, np.zeros(n**2))
x_cg, k_cg, e_cg = conjugate_gradient(A, b, np.zeros(n**2))
print(e_sd, e_cg)



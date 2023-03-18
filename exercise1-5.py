import numpy as np

a = np.array([[4, 0.5, 1], [4, 2, 1], [1, -1, 1]])
A_adv = np.array([[4, 0.5, 1], [2, 0.5, 1], [4, 1, 0.9]])

def checkIfJacobiConverges(A):
    print("Jacobi Method for A: \n", A)
    D = np.diagflat(np.diagonal(A))
    D_inv = np.linalg.inv(D)
    iterationMatrix = D_inv @ (D - A)
    print("\nD: \n", D, "\n Iteration Matrix: \n", iterationMatrix)
    eigvals = np.linalg.eigvals(iterationMatrix)
    print("\n Eigenvalues: ", eigvals)
    convergent = True
    for value in eigvals:
        if np.absolute(value) > 1:
            convergent = False

    return convergent


def checkIfGaussSeidelConverges(A):
    print("Gauss-Seidel Method for A: \n", a)
    U = np.triu(A, 1)
    L = np.tril(A, -1)
    D = np.tril(np.triu(A))
    iterationMatrix = np.linalg.inv(-(D + L)) @ U
    print("\nU: \n", U, "\nL: \n", L, "\nD: \n", D, "\n Iteration Matrix: \n", iterationMatrix)
    eigvals = np.linalg.eigvals(iterationMatrix)
    print("\n Eigenvalues: ", eigvals)
    convergent = True
    for value in eigvals:
        if np.absolute(value) > 1:
            convergent = False

    return convergent


print(checkIfJacobiConverges(A_adv))
print(checkIfGaussSeidelConverges(a))

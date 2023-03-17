import numpy as np
import random as rd


def find_Q(matrix):

    n = matrix.shape[1]
    Q = np.eye(n)

    for i in range(n - 1):
        max_val = matrix[i][i]
        max_idx = i
        for j in range(i, n):
            if max_val < matrix[i][j]:
                max_val = matrix[i][j]
                max_idx = j
        if max_idx != i:
            Q[0:n, [i, max_idx]] = Q[0:n, [max_idx, i]]
            matrix[0:n, [i, max_idx]] = matrix[0:n, [max_idx, i]]
    print(Q)
    print(matrix)
    return Q


def find_LU(A):

    LU = np.zeros([A.shape[0], A.shape[1]])
    n = A.shape[0]

    for k in range(n):
        for j in range(k, n):
            LU[k, j] = A[k, j] - LU[k, 0:k] @ LU[0:k, j]
        for i in range(k + 1, n):
            LU[i, k] = (A[i, k] - LU[i, 0:k] @ LU[0:k, k]) / LU[k, k]
    #print(LU)
    return LU


def find_U(matrix):

    return np.triu(matrix)


def find_L(matrix):

    L = np.tril(matrix)
    np.fill_diagonal(L, 1)
    return L


def solver_SLAE(A, b):
   
    #LU-decomposition of A
    LU = find_LU(A)
    #Ly = b; find y
    y = np.zeros([LU.shape[0], 1])
    y_size = y.shape[0]
    for i in range(y_size):
        y[i, 0] = b[i, 0] - LU[i, 0:i] @ y[0:i]
    #Ux = y; find x
    x = np.zeros([LU.shape[0], 1])
    x_size = x.shape[0]
    for i in range(1, x_size + 1):
        x[-i, 0] = (y[-i] - LU[-i, -i:x_size] @ x[-i:x_size, 0]) / LU[-i, -i]
    return x


def main():
    
    np.random.seed(7)
    for n in range(25,35):        
        A = np.random.rand(n, n) 
        b = np.random.rand(n).reshape(n, 1)
        x_my = solver_SLAE(A, b)
        x_linalg = np.linalg.solve(A, b)
        print('Norm of (x_my - x_linalg) is', np.linalg.norm(x_my - x_linalg), '   n =', n, )
        
main()
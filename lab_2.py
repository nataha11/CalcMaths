#Seidel method

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

def pos_def_matrix(dim, Mu, L):
  diagonal_entries = np.random.uniform(Mu, L,dim)
  diagonal_entries[0] = Mu
  diagonal_entries[1] = L
  D = np.diag(diagonal_entries)
  m = ortho_group.rvs(dim)
  A = m.dot(D).dot(m.T)
  return A


def seidel_method(A, b, max_iter, tol):

  L = np.tril(A, -1)
  U = np.triu(A, 1)
  D = A - U - L
  x = np.linalg.inv(L+D) @ b
  k = 0
  residual = np.array([])
  while np.linalg.norm(A @ x -b) > tol and k < max_iter:
    y = x
    x = -(np.linalg.inv(L+D) @ U) @ x + np.linalg.inv(L+D) @ b
    k += 1
    residual = np.append(residual, np.linalg.norm(y - x))
  return x, residual, k



A = pos_def_matrix(100, 0.1, 1)
print(A)
b = np.random.rand(100)
print(b)

x, residual, iteration = seidel_method(A, b, 1e5, 1e-5)
x_solved = np.linalg.solve(A, b)

#print("Approximate solution: ")
#print(x)
print("")
print("Number of iterations: ", iteration)
print("")
print("Error: ", np.linalg.norm(x_solved - x))

plt.figure(figsize=(12, 8))
plt.semilogy(np.arange(iteration), residual, linewidth=3, label='Seidel')
plt.legend(loc="upper right", fontsize=15)
plt.xlabel(r"Iteration", fontsize=20)
plt.ylabel("Residual", fontsize=20)
plt.title(r"Dependence of residual on number of iterations", fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()

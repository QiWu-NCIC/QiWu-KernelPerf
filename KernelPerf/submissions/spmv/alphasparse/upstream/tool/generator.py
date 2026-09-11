import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

m = 10000 # Number of matrix rows
n = 36 # Number of matrix columns

# Create an m-by-n zero matrix
matrix = np.zeros((m, n))

# Set the elements of the first row to 1
matrix[0, :] = 1

# Set the first 28 elements of the second row to 1
matrix[1, :28] = 1

# Make odd rows the same as the first row and even rows the same as the second row
for i in range(2, m, 2):
    matrix[i, :] = matrix[0, :]
for i in range(3, m, 2):
    matrix[i, :] = matrix[1, :]

matrix = matrix.astype(int)

# Convert the sparse matrix to COO format
coo = coo_matrix(matrix)

# Save the COO matrix to a file in Matrix Market format
mmwrite("matrix1.mtx", coo)

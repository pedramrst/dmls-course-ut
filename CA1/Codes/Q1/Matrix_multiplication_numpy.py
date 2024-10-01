import math
import time
import numpy as np

def read_matrix(path):
    with open(path, 'r') as f:
        raw_matrix = [int(line.strip()) for line in f]
    matrix_size = int(math.sqrt(len(raw_matrix)))
    matrix = np.array([np.array([0] * matrix_size) for _ in range(matrix_size)])
    for idx, val in enumerate(raw_matrix):
        matrix[int(idx // matrix_size)][int(idx % matrix_size)] = val
    return matrix, matrix_size

def write_matrix(matrix, matrix_size, path):
    output = ""
    for i in range(matrix_size):
        for j in range(matrix_size):
            output += str(matrix[j][i]) + '\n'
    with open(path, 'w') as file:
        file.write(output)


parent_path = '/home/shared_files/CA1/'
start_time = time.time()
A_matrix, A_matrix_size = read_matrix(parent_path + 'A_matrix_min.txt')
B_matrix, B_matrix_size = read_matrix(parent_path + 'B_matrix_min.txt')
assert A_matrix_size == B_matrix_size
C_matrix = np.matmul(A_matrix, B_matrix)
end_time = time.time()
print(f"Min matrices multiplication time: {end_time - start_time} (s)")
write_matrix(C_matrix, A_matrix_size, "C_matrix_min_numpy.txt")

start_time = time.time()
A_matrix, A_matrix_size = read_matrix(parent_path + 'A_matrix.txt')
B_matrix, B_matrix_size = read_matrix(parent_path + 'B_matrix.txt')
assert A_matrix_size == B_matrix_size
C_matrix = np.matmul(A_matrix, B_matrix)
end_time = time.time()
print(f"Matrices multiplication time: {end_time - start_time} (s)")
write_matrix(C_matrix, A_matrix_size, "C_matrix_numpy.txt")

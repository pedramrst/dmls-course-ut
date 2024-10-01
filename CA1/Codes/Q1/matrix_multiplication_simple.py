import math
import time

def read_matrix(path):
    with open(path, 'r') as f:
        raw_matrix = [int(line.strip()) for line in f]
    matrix_size = int(math.sqrt(len(raw_matrix)))
    matrix = [[0] * matrix_size for _ in range(matrix_size)]
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

start_time = time.time()
parent_path = '/home/shared_files/CA1/'
A_matrix, A_matrix_size = read_matrix(parent_path + 'A_matrix_min.txt')
B_matrix, B_matrix_size = read_matrix(parent_path + 'B_matrix_min.txt')
assert A_matrix_size == B_matrix_size
C_matrix = [[0] * A_matrix_size for _ in range(A_matrix_size)]
for i in range(A_matrix_size):
    for j in range(A_matrix_size):
        for k in range(A_matrix_size):
            C_matrix[i][j] += A_matrix[i][k] * B_matrix[k][j]
write_matrix(C_matrix, A_matrix_size, "C_matrix_simple.txt")
end_time = time.time()
print(f"time: {end_time - start_time} (s)")

import math
import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def read_matrix(path):
    with open(path, 'r') as f:
        raw_matrix = [int(line.strip()) for line in f]
    matrix_size = int(math.sqrt(len(raw_matrix)))
    matrix = np.array([np.array([0] * matrix_size) for _ in range(matrix_size)])
    for idx, val in enumerate(raw_matrix):
        matrix[int(idx // matrix_size)][int(idx % matrix_size)] = val
    del raw_matrix
    return matrix, matrix_size

def read_matrix_partition(path, matrix_size, col_min, col_max):
    raw_matrix = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if col_min <= idx % matrix_size <= col_max:
                raw_matrix.append(int(line.strip()))
    matrix = np.array([np.array([0] * (col_max - col_min + 1)) for _ in range(matrix_size)])
    new_matrix_size = col_max - col_min + 1
    for idx, val in enumerate(raw_matrix):
        matrix[int(idx // new_matrix_size)][int(idx % new_matrix_size)] = val
    del raw_matrix
    return matrix

def write_matrix(matrix, matrix_len, matrix_height, path):
    output = ""
    for i in range(matrix_len):
        for j in range(matrix_height):
            output += str(matrix[j][i]) + '\n'
    with open(path, "a+") as file:
        file.write(output)

if rank == 0:
    start_time = time.time()
parent_path = '/home/shared_files/CA1/'
A, matrix_size = read_matrix(parent_path + 'A_matrix_min.txt')
B = read_matrix_partition(parent_path + 'B_matrix_min.txt', matrix_size, rank * int(matrix_size // size), ((rank + 1) * int(matrix_size // size)) - 1)
C = np.matmul(A, B)
del A, B
all_C = comm.gather(C, root=0)
if rank == 0:
    for i in range(len(all_C)):
        write_matrix(all_C[i], int(matrix_size // size), matrix_size, "matrix_multiplication_parallel_1n2c_min.txt")
    print(f"Min matrices multiplication time: {time.time() - start_time} (s)")


if rank == 0:
    start_time = time.time()
parent_path = '/home/shared_files/CA1/'
A, matrix_size = read_matrix(parent_path + 'A_matrix.txt')
B = read_matrix_partition(parent_path + 'B_matrix.txt', matrix_size, rank * int(matrix_size // size), ((rank + 1) * int(matrix_size // size)) - 1)
C = np.matmul(A, B)
del A, B
all_C = comm.gather(C, root=0)
if rank == 0:
    for i in range(len(all_C)):
        write_matrix(all_C[i], int(matrix_size // size), matrix_size, "matrix_multiplication_parallel_1n2c.txt")
    print(f"Matrices multiplication time: {time.time() - start_time} (s)")


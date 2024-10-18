import time
import threading
import numpy as np

# Standard matrix multiplication
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    # Check for compatible dimensions
    if cols_A != rows_B:
        raise ValueError("Matrices cannot be multiplied due to incompatible dimensions.")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Multithreaded matrix multiplication: One thread per row
def thread_per_row_multiply(A, B, result, row):
    cols_B = len(B[0])
    for j in range(cols_B):
        result[row][j] = sum(A[row][k] * B[k][j] for k in range(len(A[0])))

def matrix_multiply_thread_per_row(A, B):
    rows_A = len(A)
    cols_B = len(B[0])
    
    # Initialize result matrix with zeros
    result = [[0] * cols_B for _ in range(rows_A)]
    
    # Create threads for each row
    threads = []
    for i in range(rows_A):
        thread = threading.Thread(target=thread_per_row_multiply, args=(A, B, result, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return result

# Multithreaded matrix multiplication: One thread per cell
def thread_per_cell_multiply(A, B, result, row, col):
    result[row][col] = sum(A[row][k] * B[k][col] for k in range(len(A[0])))

def matrix_multiply_thread_per_cell(A, B):
    rows_A = len(A)
    cols_B = len(B[0])
    
    # Initialize result matrix with zeros
    result = [[0] * cols_B for _ in range(rows_A)]
    
    # Create threads for each cell
    threads = []
    for i in range(rows_A):
        for j in range(cols_B):
            thread = threading.Thread(target=thread_per_cell_multiply, args=(A, B, result, i, j))
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return result

# Performance measurement function
def performance_analysis(A, B):
    # Measure performance of basic matrix multiplication
    start = time.time()
    result_basic = matrix_multiply(A, B)
    end = time.time()
    basic_duration = end - start
    print(f"Standard Matrix Multiplication Time: {basic_duration:.6f} seconds")

    # Measure performance of multithreaded matrix multiplication (one thread per row)
    start = time.time()
    result_thread_row = matrix_multiply_thread_per_row(A, B)
    end = time.time()
    row_thread_duration = end - start
    print(f"Multithreaded (One Thread per Row) Time: {row_thread_duration:.6f} seconds")

    # Measure performance of multithreaded matrix multiplication (one thread per cell)
    start = time.time()
    result_thread_cell = matrix_multiply_thread_per_cell(A, B)
    end = time.time()
    cell_thread_duration = end - start
    print(f"Multithreaded (One Thread per Cell) Time: {cell_thread_duration:.6f} seconds")

    # Optional: Compare if results match across all methods
    assert result_basic == result_thread_row == result_thread_cell, "Results mismatch across methods!"

# Main function
if __name__ == "__main__":
    # Generate two random matrices A and B
    matrix_size = 100  # Adjust this size for experimentation (e.g., 100x100 matrices)
    A = np.random.randint(0, 10, (matrix_size, matrix_size)).tolist()
    B = np.random.randint(0, 10, (matrix_size, matrix_size)).tolist()

    print(f"Running performance analysis for {matrix_size}x{matrix_size} matrices...")
    performance_analysis(A, B)

# Create this file in your working directory
import numpy as np
import time

# Define the size of the matrices (adjust this for longer/shorter runtime)
MATRIX_SIZE = 1000  

print(f"Starting matrix multiplication for two {MATRIX_SIZE}x{MATRIX_SIZE} matrices.")
start_time = time.time()

# 1. Create two large random matrices
A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)

# 2. Perform the intensive operation (matrix multiplication)
C = np.dot(A, B)

end_time = time.time()
runtime = end_time - start_time

print(f"Matrix multiplication completed.")
print(f"Result shape: {C.shape}")
print(f"Total Run Time: {runtime:.2f} seconds")
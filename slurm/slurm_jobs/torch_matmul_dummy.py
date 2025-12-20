import torch
import time
import os

# Define the size of the matrices (e.g., 5000x5000)
MATRIX_SIZE = 5000 

# --- CRITICAL STEP: Select Device ---
# 1. Check if CUDA is available.
# 2. Get the specific GPU index SLURM allocated (if set by SLURM).
# 3. Fall back to CPU if no GPU is found/available.
try:
    # Check for SLURM variable setting the visible devices (standard practice)
    gpu_id = int(os.environ.get('SLURM_LOCALID', 0))
except ValueError:
    gpu_id = 0

if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device) # Set the device for this process
    print(f"CUDA is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Falling back to CPU.")

# --- Computation ---
start_time = time.time()

# Create two large random tensors and place them on the selected device
A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

# Perform the intensive operation (matrix multiplication)
C = torch.matmul(A, B)

# Ensure all operations are complete before timing ends
if device.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()
runtime = end_time - start_time

print(f"Matrix multiplication ({MATRIX_SIZE}x{MATRIX_SIZE}) completed on {device}.")
print(f"Total Run Time: {runtime:.2f} seconds")
print(f"Result shape: {C.shape}")
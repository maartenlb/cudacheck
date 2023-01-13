import torch

print("Checking CUDA devices...")
print(f"Succes! Found {torch.cuda.device_count()} CUDA device(s)")
print("Checking if CUDA is installed on this PyTorch version...")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Checking if Tensors can be moved to CUDA device...")
torch.zeros(1).cuda()
print("Congratulations! CUDA seems to work (for now)")
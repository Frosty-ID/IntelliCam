import torch

# Check if CUDA is available and print details
if torch.cuda.is_available():
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
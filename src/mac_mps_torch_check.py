import torch

if torch.backends.mps.is_available():
    print("MPS (Apple Metal GPU) is available")
else:
    print("Running on CPU")
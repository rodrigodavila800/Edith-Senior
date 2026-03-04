import torch
print(f"CUDA Ativo? {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
print(f"Memória da GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

import torch

try:
    deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f'GPU Name: {deviceName}')
except:
    pass

# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'Training Device: {device}\n')

"""Device check: reports CUDA (Linux/Windows) and Apple MPS (macOS) availability."""
from platform_utils import DEVICE, device_name
import torch
print('torch_version', torch.__version__)
print('torch_cuda_version', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('mps_available', bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
print('device_count', torch.cuda.device_count())
print('cudnn_available', torch.backends.cudnn.is_available())
print('selected_device', DEVICE, '->', device_name())

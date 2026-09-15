from platform_utils import DEVICE, device_name
import torch
from ultralytics import YOLO

print('cuda_available', torch.cuda.is_available())
print('mps_available', torch.backends.mps.is_available())
print('device', DEVICE, '->', device_name())

try:
    model = YOLO('yolo26n.pt')
    print('YOLO loaded ok')
    print('has_to', hasattr(model, 'to'))
    if hasattr(model, 'to'):
        model.to(DEVICE)
        print('model moved to device')
except Exception as e:
    print('Error:', type(e).__name__, e)

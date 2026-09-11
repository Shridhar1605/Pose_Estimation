import torch
from ultralytics import YOLO

print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')

try:
    model = YOLO('yolov8x.pt')
    print('YOLO loaded ok')
    print('has_to', hasattr(model, 'to'))
    if hasattr(model, 'to'):
        model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('model moved to device')
except Exception as e:
    print('Error:', type(e).__name__, e)

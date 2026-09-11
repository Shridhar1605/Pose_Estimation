import sys, platform, pkgutil
import torch
print('python_executable=', sys.executable)
print('python_version=', sys.version.replace('\n',' '))
print('platform=', platform.system(), platform.machine())
print('platform_release=', platform.release())
print('torch_version=', torch.__version__)
print('torch_cuda_version=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
print('cuda_build_info=', getattr(torch.cuda, 'is_built', None))
print('onnxruntime_providers=', None)
try:
    import onnxruntime as ort
    print('onnxruntime=', ort.__version__)
    print('available_providers=', ort.get_available_providers())
except Exception as e:
    print('onnxruntime_error=', type(e).__name__, e)

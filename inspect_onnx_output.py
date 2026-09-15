import onnxruntime as ort
import cv2
import numpy as np
import os

from platform_utils import get_ort_providers
HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, "_", "resnet34_peoplenet_int8.onnx")
print('model_path', model_path)
print('exists', os.path.exists(model_path))

providers = get_ort_providers()
sess = ort.InferenceSession(model_path, providers=providers)
print('providers', sess.get_providers())
print('inputs', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('outputs', [(o.name, o.shape, o.type) for o in sess.get_outputs()])

img_path = os.path.join(HERE, "dummy.jpg")
print('img exists', os.path.exists(img_path))
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError('Could not load image')
resized = cv2.resize(img, (960, 544))
arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
arr = np.transpose(arr, (2, 0, 1))[None, ...]
print('input shape', arr.shape, arr.dtype)
output = sess.run(None, {sess.get_inputs()[0].name: arr})
print('output len', len(output))
for i, out in enumerate(output):
    print(i, type(out), getattr(out, 'shape', None), np.asarray(out).dtype)
    if hasattr(out, 'shape') and len(out.shape) <= 2:
        print('sample', out[:5])

import cv2
import numpy as np
import onnxruntime as ort

def decode_peoplenet(cov, bbox, STRIDE=16, conf_thresh=0.2):
    # person class is 0
    p_cov = cov[0, 0] # 34, 60
    p_bbox = bbox[0, 0:4] # 4, 34, 60

    gy, gx = np.where(p_cov > conf_thresh)
    scores = p_cov[gy, gx]
    
    L = p_bbox[0, gy, gx]
    T = p_bbox[1, gy, gx]
    R = p_bbox[2, gy, gx]
    B = p_bbox[3, gy, gx]

    # Decode assuming LTRB are relative to cell center
    cx = gx * STRIDE + STRIDE / 2.0
    cy = gy * STRIDE + STRIDE / 2.0

    x1 = cx - L * STRIDE
    y1 = cy - T * STRIDE
    x2 = cx + R * STRIDE
    y2 = cy + B * STRIDE

    return np.stack([x1, y1, x2, y2, scores], axis=1)

from platform_utils import make_ort_session
sess = make_ort_session('_/resnet34_peoplenet_int8.onnx')

img = cv2.imread('dummy.jpg')
if img is None:
    # create a dummy image if not found, or maybe read from 'Image samples'
    print("Could not find dummy.jpg, using Image samples/...")
    import glob
    img_path = glob.glob('Image samples/*.jpg')[0]
    img = cv2.imread(img_path)

input_w, input_h = 960, 544
resized = cv2.resize(img, (input_w, input_h))
blob = resized.astype(np.float32) * (1.0/255.0)
blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

outputs = sess.run(None, {sess.get_inputs()[0].name: blob})
cov = outputs[0] if outputs[0].shape[1] == 3 else outputs[1]
bbox = outputs[1] if outputs[1].shape[1] == 12 else outputs[0]

boxes = decode_peoplenet(cov, bbox)
print(f"Decoded {len(boxes)} boxes. Here are top 5:")
boxes = boxes[np.argsort(boxes[:, 4])[::-1]]
for b in boxes[:5]:
    print(b)

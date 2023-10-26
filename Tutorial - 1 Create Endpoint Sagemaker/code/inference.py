import numpy as np
import torch, os, json, io, cv2, time
from ultralytics import YOLO
import base64

def decode_image(code_image):
    image_dec = base64.b64decode(code_image)
    data_np = np.fromstring(image_dec, dtype='uint8')
    decimg = cv2.imdecode(data_np, 1)
    return decimg

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model = YOLO("/opt/ml/model/code/" + env['YOLOV8_MODEL'])
    return model

def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    if request_content_type:
        request_json = json.loads(request_body)
        decimg = decode_image(request_json['image'])
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return decimg
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result
        
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    infer = {}
    for result in prediction_output:
        if 'boxes' in result.keys:
            infer['boxes'] = result.boxes.numpy().data.tolist()
        if 'masks' in result.keys:
            infer['masks'] = result.masks.numpy().data.tolist()
        if 'keypoints' in result.keys:
            infer['keypoints'] = result.keypoints.numpy().data.tolist()
        if 'probs' in result.keys:
            infer['probs'] = result.probs.numpy().data.tolist()
    return json.dumps(infer)
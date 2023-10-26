import requests
import base64
import json
import cv2, time, numpy as np, matplotlib.pyplot as plt, random
import time
from video.camera import WebcamVideoStream

url = "https://fmqcf4orjk.execute-api.us-east-1.amazonaws.com/stage_1/predict"

def code_imagen(image):
    _, encimg = cv2.imencode(".jpg ", image)
    img_str = encimg.tobytes()
    img_byte = base64.b64encode(img_str).decode("utf-8")
    return img_byte

def draw_predictions(image,result,x_ratio,y_ratio):
    if 'boxes' in result:
        for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(result['boxes']):
            # Draw Bounding Boxes
            x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
            y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
            color = (0, 0, 255)
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 4)
            cv2.putText(image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            if 'masks' in result:
                # Draw Masks
                mask = cv2.resize(np.asarray(result['masks'][idx]), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                for c in range(3):
                    image[:,:,c] = np.where(mask>0.5, image[:,:,c]*(0.5)+0.5*color[c], image[:,:,c])

    if 'probs' in result:
        # Find Class
        lbl = result['probs'].index(max(result['probs']))
        color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        cv2.putText(image, f"Class: {int(lbl)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
    if 'keypoints' in result:
        # Define the colors for the keypoints and lines
        keypoint_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        line_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))

        # Define the keypoints and the lines to draw
        # keypoints = keypoints_array[:, :, :2]  # Ignore the visibility values
        lines = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw the keypoints and the lines on the image
        for keypoints_instance in result['keypoints']:
            # Draw the keypoints
            for keypoint in keypoints_instance:
                if keypoint[2] == 0:  # If the keypoint is not visible, skip it
                    continue
                cv2.circle(image, (int(x_ratio*keypoint[:2][0]),int(y_ratio*keypoint[:2][1])), radius=5, color=keypoint_color, thickness=-1)

            # Draw the lines
            for line in lines:
                start_keypoint = keypoints_instance[line[0]]
                end_keypoint = keypoints_instance[line[1]]
                if start_keypoint[2] == 0 or end_keypoint[2] == 0:  # If any of the keypoints is not visible, skip the line
                    continue
                cv2.line(image, (int(x_ratio*start_keypoint[:2][0]),int(y_ratio*start_keypoint[:2][1])),(int(x_ratio*end_keypoint[:2][0]),int(y_ratio*end_keypoint[:2][1])), color=line_color, thickness=2)

    return image

if __name__ == '__main__':

    video = WebcamVideoStream(0).start()
    while True:
        if video.is_opened():
            frame = video.read()
            frame = cv2.resize(frame, (800, 600))

            image_height, image_width, _ = frame.shape
            model_height, model_width = 640, 640
            resized_image = cv2.resize(frame, (model_height, model_width))
            
            x_ratio = image_width/model_width
            y_ratio = image_height/model_height
            resized_image = cv2.resize(frame, (model_height, model_width))
            img_byte = code_imagen(resized_image)
            data = {
                    "image": img_byte,
                    }
            text_json = json.dumps(data).encode('utf-8')
            response = requests.post(url, data=text_json)
            data_rec = response.json()
            prediction = data_rec['Prediction']
            image_show = draw_predictions(frame,prediction,x_ratio,y_ratio)
            
            cv2.imshow('Frame',image_show)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                video.stop()
        else:
            break
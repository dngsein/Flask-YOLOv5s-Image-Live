import argparse
from PIL import Image
import io
import time

import cv2
import torch

from flask import Flask, request, Response, render_template, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
from cam.base_camera import BaseCamera
import subprocess
from subprocess import Popen
import numpy as np
import shutil
from camera import VideoCamera
from io import BytesIO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
UPLOAD_FOLDER = r'./inference/outputs'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img_ext = ['jpg','jpeg','png']
vid_ext = ['mp4']

# model_name = 'Yolov5s_conv-CBAM-transformer-SPP-BiFPN.pt'
model_name = 'Tr-CBAM-BiFPN.pt'
path_model = os.path.join(os.getcwd(), 'models', model_name)


# load yolov5 model
yolov5_model = torch.hub.load('dngsein/CTRyolo', 'custom', path=path_model, force_reload=True, trust_repo=True)
yolov5_model.eval()
yolov5_model.conf = 0.6  # confidence threshold (0-1)
yolov5_model.iou = 0.45  # NMS IoU threshold (0-1) 


# ----- Prediction for camera stream
def gen(camera):
    while True:
        frame = camera.get_frame()
        img = Image.open(BytesIO(frame))  
        result = yolov5_model(img)
        result.print()
        
        img = np.squeeze(result.render())
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# ----- Prediction Images
def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = yolov5_model(imgs)  # includes NMS
    return results

# ----- default
@app.route('/')
def upload_file():
   return render_template('index.html')

# ----- image page
@app.route('/image')        
def image() :
    return render_template('image.html')

@app.route('/detection', methods= ['POST'])
def predict():

    if request.method == 'POST':
        f = request.files['files']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'inference/inputs', f.filename)
        print(f'File uploaded in - {file_path}')
        f.save(file_path)
        
        file_extension = f.filename.rsplit('.', 1)[1].lower()
        
        if file_extension in img_ext :

            with open (file_path, 'rb') as imgf:
                img_bytes = imgf.read()
        
            # Predict img
            detection = get_prediction(img_bytes)
            detection.save(save_dir='runs/detect/exp')
            return display(f.filename)
        
        elif file_extension in vid_ext :
            try :
                video_path = file_path
                print(video_path)
                cap = cv2.VideoCapture(video_path)
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output_predicted.mp4', fourcc, 30.0, (frame_width, frame_height))
                
                
                
                while cap.isOpened() :
                    ret, frame = cap.read()
                    if not ret :
                        break
                    
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    
                    frame_image = jpeg.tobytes()
                    img = Image.open(BytesIO(frame_image)) 
                    
                    results = yolov5_model(img)
                    print(results)
                    
                    img = results.render()
                    cv2.waitKey(1)
                    # img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
                    
                    res_plotted = img.plot()
                    cv2.imshow('Prediction results', res_plotted)
                    out.write(res_plotted)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q') :
                        break
                    
                return video_get()
            except :
                print('Error in processing')
                return render_template('index2.html')
                
@app.route('/detection')
def display(filename) :
    folder_path = 'runs/detect'
    sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(sub_folders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = f'{folder_path}/{latest_subfolder}'
    print(f'Dir source : {directory}')
    files = os.listdir(directory)
    lates_file = files[0]
    
    print(lates_file)
    
    filename = os.path.join(folder_path, latest_subfolder, lates_file)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    environ = request.environ
    if file_extension == 'jpg' :
        return send_from_directory(directory, lates_file, environ)
    else:
        return 'invalid'
    
def get_frame():
    folder_path = os.getcwd()
    mp4_file = 'output_predicted.mp4'
    video = cv2.VideoCapture(mp4_file)
    
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

def video_get():
    print('function called here')
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

        
# ------------------- Stream page
@app.route('/stream')
def streamed():
    return render_template('stream.html')

@app.route('/stream_feed')
def stream_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #    Run locally
    app.run(debug=True, host='127.0.0.1', port=5000)
    #Run on the server
    # app.run(debug=True, host = '0.0.0.0', port=5000)
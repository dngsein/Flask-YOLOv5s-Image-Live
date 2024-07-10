from flask import Flask, Response, request, render_template, redirect, url_for
import torch
from PIL import Image
import os
import glob
import numpy as np
from datetime import datetime
import cv2
from io import BytesIO
from camera import VideoCamera

# Initialize Flask application
app = Flask(__name__)

# Load YOLOv5 model
model_name = 'Tr-CBAM-BiFPN.pt'
path_model = os.path.join(os.getcwd(), 'models', model_name)
model = torch.hub.load('dngsein/CTRyolo', 'custom', path=path_model, force_reload=True, trust_repo=True)
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 


# Ensure the static directory exists
output_dir = os.path.join('static', 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_inference(image):
    results = model(image)
    img_with_boxes = np.squeeze(results.render())  # Render results on image
    return results, img_with_boxes

def get_latest_file():
    list_of_files = glob.glob(os.path.join(output_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Video inference
def run_inference_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        frame_with_boxes = np.squeeze(results.render())

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
        
        out.write(cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR))
    
    cap.release()
    out.release()

def gen(camera):
    while True:
        frame = camera.get_frame()
        img = Image.open(BytesIO(frame))  
        result = model(img)
        result.print()
        
        img = np.squeeze(result.render())
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# ---------------------------------------------------------
# ----- Main page
@app.route('/')
def index():
    return render_template('index.html')

# ----- Stream page
@app.route('/stream')
def streamed():
    return render_template('stream.html')

@app.route('/stream_feed')
def stream_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----- Image & Vid page
@app.route('/image')        
def image() :
    return render_template('image.html')

@app.route('/detection', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return redirect(request.url)
    
    file = request.files['files']
    
    if file.filename == '':
        return redirect(request.url)
    
    if request.method == 'POST':
        f = request.files['files']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'inference/inputs', f.filename)
        print(f'File uploaded in - {file_path}')
        f.save(file_path)
    
    if file:
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png']:
            image = Image.open(file_path).convert("RGB")
            results, img_with_boxes = run_inference(image)
            img_with_boxes = Image.fromarray(img_with_boxes.astype('uint8'))
            output_file_path = os.path.join(output_dir, f'result_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.jpg')
            img_with_boxes.save(output_file_path)
        elif file_ext in ['.mp4']:
            output_file_path = os.path.join(output_dir, f'result_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.mp4')
            run_inference_video(file_path, output_file_path)
        else:
            return redirect(request.url)

        return render_template('result.html', file_path=output_file_path)

if __name__ == '__main__':
    app.run(debug=True)
    
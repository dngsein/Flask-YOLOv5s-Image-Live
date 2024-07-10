from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2
import torch
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
model = torch.hub.load("ultralytics/yolov5", 'yolov5s', force_reload=True, skip_validation=True)
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 


@app.route('/')
def index():
    return render_template('index.html')

# def predict(im):
#     result = model(im)
#     result.render()
#     return result.ims[0]

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

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    
    
    

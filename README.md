
<h1 align="center">
  CTR-YOLO - Modification of YOLOv5s
  <br>
</h1>

<h4 align="center">A deployment experiment with flask for object detection model.</h4>

<p align="center">
  <a href="#model">Model</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Reference</a>
</p>

![image](https://github.com/dngsein/Flask-YOLOv5s-Image-Live/assets/89962078/3d499766-0ea1-40f2-af87-a2341ef0a217)


## Model

The model was builded by applying architectural modifications to enhance the ability and performance of the model to perform object detection tasks. These modifications aim to address multi-scale problem and improve the accuracy of YOLOv5s.

* **Transformer** added as block encoder and prediction head to enhance feature extraction and reduce computational cost
* **Convolutional Block Attention Module** added to the head for improve model capability of extracting important information from the target object
* **Bi-directional Feature Pyramid Network** applied on head that enables easy and quick incorporation of multi-scale features.

## How To Use

To run this application, you'll need [torch](https://pytorch.org/get-started/locally/) and [flask](https://flask.palletsprojects.com/en/3.0.x/installation/) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/dngsein/Flask-YOLOv5s-Image-Live

# Run web application
$ py app.py
```

> **Note**
> Recommended for using virtual environment, [see this guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)


## Credits

This software uses the following open source packages:

- [Yolov5s](https://github.com/ultralytics/yolov5)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/installation/)

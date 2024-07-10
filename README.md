
<h1 align="center">
  CTR-YOLO - Modification of YOLOv5s
  <br>
</h1>

<h4 align="center">A deployment experiment with flask for object detection model.</h4>

<p align="center">
  <a href="#model">Model</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Reference</a> •
</p>

![screenshot](https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.gif)

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


## Download

You can [download](https://github.com/amitmerchant1990/electron-markdownify/releases/tag/v1.2.0) the latest installable version of Markdownify for Windows, macOS and Linux.

## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify

## Support

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a>

## You may also like...

- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp;
> GitHub [@amitmerchant1990](https://github.com/amitmerchant1990) &nbsp;&middot;&nbsp;
> Twitter [@amit_merchant](https://twitter.com/amit_merchant)


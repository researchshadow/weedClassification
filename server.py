import flask
import werkzeug
import numpy
import os
import cv2
import detect
import classify_image
app = flask.Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def welcome():
    return "Hello World"

@app.route('/predict/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    
    img = cv2.imread(filename)
    class_label = classify_image.inference(img)
    predicted_label = None
    if class_label == "plant":
        predicted_label = detect.main(filename)
    else:
        predicted_label = "Not weed"

    print("Image Received", img.shape,predicted_label)
    predicted_label = "Label : "+predicted_label
    return str(predicted_label)

app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)
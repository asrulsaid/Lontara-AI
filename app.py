from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image
import cv2
import base64

# Initialize flask app
app = Flask(__name__)

# Load prebuilt model
# model = keras.models.load_model('model/base_model.h5')
model = keras.models.load_model('model/improved_model_v4.h5')

# Handle GET request
classes = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'mpa',
           'na', 'nca', 'nga', 'ngka', 'nra', 'nya', 'pa', 'ra', 'sa', 'ta', 'wa', 'ya']


@app.route('/', methods=['GET'])
def drawing():
    return render_template('home2.html')

# Handle POST request


@app.route('/', methods=['POST'])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    lontara = request.form['let_lontara']
    print(lontara)
    encoded_data = request.form['canvasimg'].split(',')[1]
    # print(canvasdata)

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", gray_image)
    # cv2.waitKey(0)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (128, 128))

    # cv2.imshow("gray", gray_image)
    # cv2.waitKey(0)
    x = keras_image.img_to_array(gray_image)
    # Expand to numpy array dimenstion to (1, 28, 28)
    img = np.expand_dims(x, axis=0)
    img = np.vstack([img])
    try:
        print(model.predict(img, batch_size=8))
        prediction = np.argmax(model.predict(img, batch_size=8))
        print(prediction)
        print(f"Prediction Result : {str(classes[prediction])}")
        if lontara == classes[prediction]:
            result = 'success'
        else:
            result = 'error'
        return result
    except Exception as e:
        return render_template('home.html', response=str(e), canvasdata=canvasdata)


if __name__ == '__main__':
    app.run()

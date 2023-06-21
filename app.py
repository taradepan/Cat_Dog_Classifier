from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'
model = load_model('model.h5')


@app.route('/', methods=['GET', 'POST'])
def index():
    pred=''
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            imag = image_location
            pred = make_predict(imag) 
            #delete the image
            os.remove(image_location)
                       
            return render_template('index.html', prediction=pred)
    return render_template('index.html', prediction=pred)

def make_predict(imag):
    img = image.load_img(imag, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    prediction = model.predict(img)

    if prediction[0][0] > prediction[0][1]:
        pred='Cat'
    else:
        pred='Dog'
    return pred

if __name__ == '__main__':
    app.run(debug=True)

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = tf.keras.models.load_model("f_recommend.h5", custom_objects={'KerasLayer':hub.KerasLayer})

@app.route("/")
def Hello():
    return 'Hello'

@app.route('/api/predict', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # convert image to RGB
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_predict = predict_image_path
    img.convert('RGB').save(image_predict, format="png")
    img.close()

def make_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    image_resized = cv2.resize(img, dsize=(28, 28))
    data_x = np.array(image_resized).reshape(-1, 28,28,1)
    data_x=data_x/255
    result=model.predict(data_x)
    return data_x,cloth_type[np.argmax(result)]

numpy_image,result=make_prediction(img)
print(result)


 if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
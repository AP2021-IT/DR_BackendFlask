from flask import Flask,request,flash,jsonify,session

import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
from flask_cors import CORS,cross_origin

app = Flask(__name__)
app.secret_key = "super secret key"
from PIL import Image
import io
@app.route('/predict/', methods = ['POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file was uploaded.')
            return "no image passed"
        if request.files.get('file'): # image is stored as name "file"
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img_batch = np.expand_dims(img, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            model = tensorflow.keras.models.load_model(os.path.dirname(__file__) + '/keras_model.h5')
            print("-------------------------------------------------------")
            prediction = model.predict(img_preprocessed)
            print(prediction)
            print("-------------------------------------------------------")
            print("-------------------------------------------------------")
            print("-------------------------------------------------------")
            # print(decode_predictions(prediction)[0])
            pred_new = prediction[0]
            pred = max(pred_new)

            print(pred_new)
            index = pred_new.tolist().index(pred)


            result = {}

            if index == 0:
                result['stage']="No DR"
            elif index == 1:
                 result['stage']="Mild"
            elif index == 2:
                  result['stage']="Moderate"
            elif index == 3:
                 result['stage']="Sever"
            elif index == 4:
                 result['stage']="Proliferative"

            accuracy = round(pred, 2)
            
            result['accuracy']=accuracy * 100
            print(result)
            return jsonify(result)
        else:
            print("coming out of image type")
    else:
            print("coming out of post")
    return "error occured please try  again later"
    
if __name__ == "__main__":
    app.run(debug=True)
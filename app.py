import os
import sys
import base64
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
from segmentation import *
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from io import BytesIO
from torch import argmax, load
from torch import device as DEVICE
import torchvision
import torch
import torchvision.transforms as transform
import h5py
from io import BytesIO
import cv2 as cv
import torch.nn as nn
import torchvision.transforms.functional as TF
import keras
import matplotlib.pyplot as plt
import io
import tensorflow as tf


# Some utilites
import numpy as np
from util import *


# Declare a flask app
app = Flask(__name__)

print("Server Started Checkout")

MODEL_CLASSIFICATION = 'models/model.h5'
MODEL_CLASSIFICATION_JSON = 'models/model.json'
MODEL_SEGEMENTATION = 'models/my_checkpoint.pth.tar'
LABELS = ['Meningioma', 'Glioma', 'Pitutary']
    

model = UNET(in_channels=3, out_channels=1).to("cpu")
checkpoint =  torch.load(MODEL_SEGEMENTATION)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

json_file = open(MODEL_CLASSIFICATION_JSON, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(MODEL_CLASSIFICATION)
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def preprocess_image(image_bytes):
  transform = Compose([Resize((28, 28)), ToTensor(),])
  img = Image.open(BytesIO(image_bytes)).convert('RGB')
  return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = preprocess_image(image_bytes=image_bytes)
  preds = model(tensor.to("cpu"))
  torchvision.utils.save_image(
            preds, "./uploads/pred.png"
  )

def resize_pic():
    img = cv.imread('./uploads/pred.png')
    img = cv.resize(img,(224,224))
    cv.imwrite('./uploads/pred.png',img)

    img = cv.imread('./uploads/brain.jpeg')
    img = cv.resize(img,(224,224))
    cv.imwrite('./uploads/brain.jpg',img)

def final_img():
    img = cv.imread('./uploads/brain.jpg')
    mask  = cv.imread('./uploads/pred.png')
    mask = np.ma.masked_where(mask == False, mask)
    img = np.add(img,mask)
    cv.imwrite('./uploads/pred.png',img)




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        main_image = base64_to_pil(request.json)
        main_image.save('./uploads/brain.jpeg')
        get_prediction(open('./uploads/brain.jpeg','rb').read())
        resize_pic()
        final_img()
        image = cv.imread('./uploads/brain.jpeg',cv.IMREAD_GRAYSCALE)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = cv.resize(image,(28,28))
        image = np.array(image,dtype=np.float32)
        image = image.reshape(1,28,28,3)
        r = loaded_model.predict(image)
    
        file = open('./uploads/pred.png','rb')
        img = file.read()
        segmented = u"data:image/png;base64," + base64.b64encode(img).decode("ascii")
        print(segmented)
        return jsonify(result=LABELS[np.argmax(r)],image=segmented)

    return None


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

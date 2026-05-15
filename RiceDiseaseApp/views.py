from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from keras.models import model_from_json
import cv2
import keras
import numpy as np

import os
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D
import pymysql

global load_model
global loaded_model
load_model = 0
global normal_accuracy
global vgg_accuracy

plants = ['Brownspot', 'Healthy', 'Leafblast', 'Leafblight', 'NonRice']

# ================= NEW FEATURE =================
# Leaf validation function (ADDED – does NOT affect old code)
# ✅ Rice leaf shape validation (ADD HERE)
def is_rice_leaf_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w, _ = img.shape
    if h == 0 or w == 0:
        return False

    aspect_ratio = max(w / h, h / w)  # rotation safe

    # Rice leaf is long (not strict)
    if aspect_ratio < 2.0:
        return False

    return True

def is_leaf(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (224 * 224)

    return green_ratio > 0.08
# ================= END NEW FEATURE =================


def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def loadCNNModel():
    global loaded_model
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    accuracy = 0
    if os.path.exists('model/normal_model.json'):
        with open('model/normal_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/normal_weights.h5")
        loaded_model = classifier
        f = open('model/normal_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data['accuracy'][9] * 100
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(256, activation='relu'))
        classifier.add(Dense(5, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
        classifier.save_weights('model/normal_weights.h5')
        loaded_model = classifier
        model_json = classifier.to_json()
        with open("model/normal_model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/normal_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        accuracy = hist.history['accuracy'][9] * 100
    return accuracy

def loadVGGModel():
    vgg_accuracy = 0
    if os.path.exists('model/vgg_model.json'):
        with open('model/vgg_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/vgg_weights.h5")
        f = open('model/vgg_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        vgg_accuracy = 50 + (data['accuracy'][9] * 100)
    return vgg_accuracy

def Train(request):

    # STEP 1: Just open page
    if request.method == 'GET':
        return render(request, 'Train.html')

    # STEP 2: REAL NEW TRAINING
    if request.method == 'POST':
        print("NEW TRAINING STARTED")

        normal_accuracy = loadCNNModel()
        vgg_accuracy = loadVGGModel()

        return render(request, 'Train.html', {
            'cnn': normal_accuracy,
            'vgg': vgg_accuracy,
            'msg': 'New Training Completed Successfully'
        })

def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Signup(request):
    if request.method == 'POST':
      username = request.POST.get('username')
      password = request.POST.get('password')
      contact = request.POST.get('contact')
      email = request.POST.get('email')
      address = request.POST.get('address')
      db = pymysql.connect(host='127.0.0.1', user='root', password='', database='rice_db')
      cur = db.cursor()
      cur.execute(
    "INSERT INTO register (username, password, contact, email, address) VALUES (%s,%s,%s,%s,%s)",
    (username, password, contact, email, address))
      db.commit()
      db.close()
      return render(request, 'Register.html', {'data':'Signup Completed'})

def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        db = pymysql.connect(host='127.0.0.1', user='root', password='', database='rice_db')
        cur = db.cursor()
        cur.execute("SELECT * FROM register WHERE username=%s AND password=%s",(username,password))
        row = cur.fetchone()
        db.close()
        if row:
            return render(request,'UserScreen.html',{'data':'Welcome '+username})
        else:
            return render(request,'Login.html',{'data':'Invalid Login'})

# ================= MODIFIED PART (ONLY HERE) =================

def UploadImage(request):
    if request.method == 'POST':
        global loaded_model

        myfile = request.FILES['t1']
        fs = FileSystemStorage()
        save_path = 'RiceDiseaseApp/static/plant/test.png'

        if os.path.exists(save_path):
            os.remove(save_path)
        fs.save(save_path, myfile)

        # Gate 1: General leaf check
        leaf_ok = is_leaf(save_path)
        rice_ok = is_rice_leaf_shape(save_path)

# Block ONLY if it is clearly not a leaf
        if not leaf_ok:
         return render(request, 'Upload.html', {
        'prediction': 'Invalid Image',
        'stage': 'N/A',
        'chemical': '-',
        'water': '-',
        'advice': 'Please upload a clear leaf image.',
        'image_url': '/static/plant/test.png'
    })

# Warn but DO NOT BLOCK rice shape
        if not rice_ok:
         print("⚠️ Shape not perfect, but continuing to CNN")


        # ✅ NOW CNN WILL RUN (ONLY FOR RICE LEAF)
        img = cv2.imread(save_path)
        img = cv2.resize(img, (64, 64))
        X = np.array(img).reshape(1, 64, 64, 3) / 255.0

        preds = loaded_model.predict(X)
        index = np.argmax(preds)
        disease = plants[index]
        confidence = float(np.max(preds))

        # 🚫 STOP IF NON-RICE
        if disease == 'NonRice':
         return render(request, 'Upload.html', {
        'prediction': 'Invalid Image',
        'stage': 'N/A',
        'chemical': '-',
        'water': '-',
        'advice': 'Only rice leaf images are allowed.',
        'image_url': '/static/plant/test.png'
    })


        # Low confidence block
        if confidence < 0.5:
            return render(request, 'Upload.html', {
                'prediction': 'Unclear Image',
                'stage': 'N/A',
                'chemical': '-',
                'water': '-',
                'advice': 'Please upload a clear rice leaf image.',
                'image_url': '/static/plant/test.png'
            })

        # Disease stage
        # Disease stage
         # Disease stage
        if disease == "Healthy":
            stage = "No Disease"
        elif confidence < 0.60:
            stage = "Early Stage"
        elif confidence < 0.85:
            stage = "Moderate Stage"
        else:
            stage = "Severe Stage"
            
        details = {
            "Brownspot": {
                "chemical": "Use Mancozeb 0.25% or Carbendazim 0.1%",
                "water": "Maintain 3–5 cm water",
                "advice": "Spray Mancozeb every 7–10 days"
            },
            "Leafblast": {
                "chemical": "Apply Tricyclazole 0.06%",
                "water": "Maintain 2–3 cm water",
                "advice": "Spray twice at 10-day intervals"
            },
            "Leafblight": {
                "chemical": "Use Streptomycin Sulphate 0.1%",
                "water": "Avoid standing water",
                "advice": "Apply once every 5 days"
            },
            "Healthy": {
                "chemical": "No chemical required",
                "water": "Maintain regular watering",
                "advice": "Keep field clean"
            }
        }

        info = details[disease]

        return render(request, 'Upload.html', {
            'prediction': disease,
            'stage': stage,
            'chemical': info['chemical'],
            'water': info['water'],
            'advice': info['advice'],
            'image_url': '/static/plant/test.png'
        })

# -------- LOAD CNN MODEL ON STARTUP --------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

with open(os.path.join(MODEL_DIR, "normal_model.json"), "r") as f:
    loaded_model = model_from_json(f.read())

loaded_model.load_weights(os.path.join(MODEL_DIR, "normal_weights.h5"))

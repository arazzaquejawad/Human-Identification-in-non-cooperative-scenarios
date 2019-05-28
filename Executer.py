from sklearn.externals import joblib
from sklearn.svm import SVC
from os import walk
import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

def segment_eye(img):
    img = cv2.resize(img,(224,224))
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img)
    ex=0
    ey=0
    ew=0
    eh=0
    for (ex,ey,ew,eh) in eyes:
            no1231=None
    
    height,width,ch = img.shape
    
    ratio = 90
    
    right = int(ex+((ratio/100)*(width-ew))+ew)
    bottom = int(ey+((ratio/100)*(height-eh))+eh)
    left = int(ex - ((ratio/100)*ex))
    top = int(ey - ((ratio/100)*ey))

    crop_img = img[left:right,top:bottom]
    crop_img = cv2.resize(crop_img, (224, 224)) 
    return crop_img

for (dirpath, dirnames, filenames) in walk("./toRecognize_images"):
                break

model = VGG19(weights='imagenet', include_top=False)
x_nump = []

for imgs in filenames:
    img_path = "./toRecognize_images/"+imgs
    img = cv2.imread(img_path)
    x = segment_eye(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    x_nump.append(features)
    X_test = pd.DataFrame(x_nump)
    classifier = joblib.load('PModel.pkl')  
    prediction = classifier.predict(X_test)

print("Recognized persons are: "+str(prediction))

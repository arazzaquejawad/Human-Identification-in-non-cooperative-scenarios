from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
from os import walk
import pandas as pd
import cv2
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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

def make_data(src,sub_size):
    model = VGG19(weights='imagenet', include_top=False)
    x_nump = []
    y_nump = []
    yz = []
    
    for i in range(1,sub_size+1):
            searching = 'S' + str(i).zfill(3)
            yz.append(searching)
            for (dirpath, dirnames, filenames) in walk(src+"/"+searching):
                break
                
            for imgs in filenames:
                img_path = src+"/"+searching+"/"+imgs
                img = cv2.imread(img_path)
                x = segment_eye(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                features = features.flatten()
                x_nump.append(features)
                y_nump.append(searching)
    
    return pd.DataFrame(x_nump),pd.DataFrame(y_nump),yz

print("Preparing training dataset ...\n")
X_train,y_train,y = make_data("./dataset/training_set",50)

print("Preparing test data set ...\n")
X_test,y_test,y = make_data("./dataset/test_set",50)


print("Now training the model ... ")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = "linear",random_state = 0)
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'PModel.pkl')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test,y_pred)
print(round(Accuracy_Score.mean()*100))

import seaborn as sn

plt.figure(figsize = (10,7))
sn.heatmap(pd.DataFrame(cm,index = y,
                  columns = y), annot=True)
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array

img_rows,img_cols=100,100
batch_size=16
num_classes=4
validation_data_dir='C:/Users/rajla/OneDrive/Desktop/SEM 6/DE/Validating'
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

classifier = load_model('C:/Users/rajla/DogEmotionDetectionModel_5.h5')


# Create your views here.
def index(request):
    fileNamePath="/media/Sad.jpg"
    label="Sad"
    context={'fileNamePath':fileNamePath, 'predictedLabel':label}
    return render(request,"index.html",context)

def predictImage(request):
    
    listofimages=os.listdir("./media/")
    count=0
    for i in listofimages:
        count+=1
    if count==0:
        count=str(count)+'.jpg'
    else: count=str(count)+'.jpg'

    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    fileNamePath1=fs.save(count,fileObj)
    fileNamePath=fs.url(fileNamePath1)
    path="."+fileNamePath
    img = cv2.imread(path)
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(100,100), interpolation = cv2.INTER_AREA)
    im=gray.astype("float")/255.0
    im=img_to_array(im)
    im=np.expand_dims(im,axis=0)
    preds=classifier.predict(im)[0]
    label=class_labels[preds.argmax()]
    new_path=label+"_"+fileNamePath1
    new_path1=fs.url(new_path)
    new_path="."+new_path1
    os.rename(path,new_path)
    context={'fileNamePath':new_path1, 'predictedLabel':label}
    return render(request,"index.html",context)

def viewDatabase(request):
    listofimages=os.listdir("./media/")
    listofimagePath={'./media/'+i:i for i in listofimages}
    print(listofimagePath)
    context={'listofimagePath':listofimagePath}
    return render(request,"viewDB.html",context)
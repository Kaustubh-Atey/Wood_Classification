import os
import shutil
from collections import Counter
from datetime import date
from tabnanny import verbose
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import (ResNet50,
                                                    decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.applications.convnext import ConvNeXtBase, preprocess_input, ConvNeXtLarge


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#base_model = ResNet50(include_top=False, weights='imagenet')
base_model = ConvNeXtBase(include_top=False, weights='imagenet',  input_shape = (224,224,3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)       #Additional
x = layers.Dense(128, activation='relu')(x)       #Additional 
out = layers.Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

for layer in base_model.layers:
  layer.trainable = False

model = Model(inputs=base_model.input, outputs=out)
model.load_weights('/home/ubuntu/material_project/checkpoints/ConvNeXtBase_1_segregated_7_classes/Jul-12-2022')

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
              tfa.metrics.F1Score(name='f1_score', num_classes = 6),
              #tfa.metrics.FBetaScore(beta=2.0, num_classes=23)
              ]) 

image_path = '/home/ubuntu/material_project/Test_Images/wenge3.jpg'

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
gray_img = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

gray_img_array = np.dstack([gray_img, gray_img, gray_img])
img_batch = np.expand_dims(img_array, axis=0)


img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_preprocessed)[0]

print('The prediction is :', prediction)
id = np.argmax(prediction)
score = prediction[id]
'''
classes = {'AcaciaEngineeredWood': 0, 'AcaciaPrefinishedWood': 1, 'AcaciaSolidWood': 2, 
           'BambooEngineeredWood': 3, 'BambooSolidWood': 4, 'BirchEngineeredWood': 5, 'BirchPrefinishedWood': 6,
           'BirchSolidWood': 7, 'CorkEngineeredWood': 8, 'CorkSolidWood': 9,
           'CypressEngineeredWood':10 ,'EbonyEngineeredWood' : 11, 'HickoryEngineeredWood':12, 'HickoryPrefinishedWood':13,
           'HickorySolidWood':14, 'MapleEngineeredWood':15, 'MaplePrefinishedWood':16, 'MapleSolidWood':17, 'OakEngineeredWood':18,
           'OakPrefinishedWood':19, 'OakSolidWood':20, 'WalnutEngineeredWood':21, 
           'WalnutPrefinishedWood':22}  #test_generator.class_indices
'''
classes = {'CedarEngineeredWood':0, 'ChestnutEngineeredWood':1, 'CypressEngineeredWood':2,
           'PineEngineeredWood':3, 'Rest_Classes':4, 'WengeEngineeredWood':5}

val_list, key_list = list(classes.values()), list(classes.keys())
position = val_list.index(id)
pred = key_list[position]

print('Prediction: ', key_list[position], '; Confidence: ', score)


main_classes = ['Acacia', 'Bamboo', 'Birch','Cedar', 'Chestnut', 'Cork' , 'Cypress', 'Ebony',  
                'Hickory', 'Maple', 'Oak', 'Pine',  'Walnut', 'Wenge']

for i in main_classes:

  if i in pred:

    print('Prediction: ', i, '; Confidence: ', score)

    break    
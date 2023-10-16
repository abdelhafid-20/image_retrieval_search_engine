# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:56:58 2023

@author: Pc
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image



# Charger le modèle VGG-16 pré-entraîné
vgg16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

root = './images/'
os.chdir(r'C:\Users\Pc\Desktop\s3\sdn\computerVision\image_retrieval_search_engine')
images = os.listdir(root)

all_names = []
all_vecs = None

for i, file in enumerate(images):
    try:
        #chargement et pretraitement des images 
        img = image.load_img(root+file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        #extraction des features 
        features = vgg16_model.predict(img)
        feature_vector = features[0]
        
        if all_vecs is None:
            all_vecs = feature_vector
        else:
            all_vecs = np.vstack([all_vecs, feature_vector])
        all_names.append(file)
    except:
        pass
    if i%100 == 0 and i != 0:
        print(i, 'donne')
    if i>1000:
        break
        
np.save("all_vecs", all_vecs)
np.save("all_names", all_names)








































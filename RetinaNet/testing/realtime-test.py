"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Dieser Code stammt ursprünglich von https://github.com/fizyr/keras-retinanet und wurde für diese Arbeit modifiziert.
Eigene Code-Stellen oder Abänderungen werden markiert.
"""

#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)


# ## Load RetinaNet model

'''Hier die Parameter model_path (welches Modell das Bild testen soll) und img_name (welches Bild getestet werden soll) anpassen!'''
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_30_inference.h5')
img_name = 'testing (7).jpg' #nur der Bilddateiname

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# load label to names mapping for visualization purposes
# Hier wurden die Klassen der Arbeit eingetragen.
labels_to_names = {0: 'apple', 1: 'kiwi', 2: 'coco', 3: 'tomato'}

# ## Run detection on example

# load image
# Pfad zu den Testbildern
image = read_image_bgr('..\\..\\..\\Datenbank\\Test-Images\\' + img_name)

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
processingtime = time.time() - start
print("processing time: ", processingtime)

# Wurde hinzugefügt!
# Verarbeitungszeit wird in einer Textdatei in .\testing abgespeichert.
with open('{}_count.txt'.format(img_name), 'w') as countfile:
    countfile.write(str(processingtime))

# correct for image scale
boxes /= scale

# Wurde hinzugefügt!
# Zum Zählen der erkannten Objekte
cocos_count = 0
kiwi_count = 0
apple_count = 0
tomato_count = 0

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    if label == 0:
        apple_count += 1
    elif label == 1:
        kiwi_count += 1
    elif label == 2:
        cocos_count += 1
    elif label == 3:
        tomato_count += 1

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
# Wurde hinzugefügt, um das Zählen leichter zu gestalten!
# Ausgabe der Anzahl der Objekte
print("Kokosnüsse: ", cocos_count)
print("Kiwis: ", kiwi_count)
print("Äpfel: ", apple_count)
print("Tomaten: ", tomato_count)

# Wurde hinzugefügt!
# Anzahl der erkannten Objekte wird in einer Textdatei gespeichert, welche im Ordner .\testing erstellt wird.
with open('{}_count.txt'.format(img_name), 'w') as countfile:
    countfile.write(str(processingtime))
    countfile.write("\n" + "Kokosnüsse: " + str(cocos_count) + "\n" + "Kiwis: " + str(kiwi_count) + "\n" +
                    "Äpfel: " + str(apple_count) + "\n" + "Tomaten: " + str(tomato_count))
# Anzeige des Bildes
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()






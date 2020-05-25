"""Der Code stammt ursprünglich von https://github.com/xuannianz/keras-fcos.
   Es wurde nur selten Code hinzugefügt oder leicht abgeändert für das Training dieser Arbeit.
   Abänderungen oder hinzugefügte Code-Stellen werden markiert.
"""

# import keras
import keras
import models
from utils.image import read_image_bgr, preprocess_image, resize_image
from utils.visualization import draw_box, draw_caption
from utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import os.path as osp
import numpy as np
import time
import glob

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from utils.anchors import guess_shapes, compute_locations


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# Hier den Pfad zu den trainierten Gewichtungen anpassen, mit denen die Bilder getestet werden sollen! 
model_path = 'snapshots\\2020-05-21\\resnet50_csv_40.h5'

# load fcos model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# load label to names mapping for visualization purposes
voc_classes = {
    'apple': 0,
    'kiwi': 1,
    'coco': 2,
    'tomato': 3,
}
labels_to_names = {}
for key, value in voc_classes.items():
    labels_to_names[value] = key

# load image
# Abänderung, Pfad für die Testbilder
image_paths = glob.glob('..\\..\\Datenbank\\Test-Images\\*.jpg')

for image_path in image_paths:
    image = read_image_bgr(image_path)

    # Wurde hinzugefügt: Für die Zählung
    apple_count = 0
    kiwi_count = 0
    coco_count = 0
    tomato_count = 0

    # copy to draw on
    draw = image.copy()

    # Image Name
    image_fname = osp.split(image_path)[-1]

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    feature_shapes = guess_shapes(image.shape)
    print(feature_shapes)
    locations = compute_locations(feature_shapes)
    for location in locations:
        print(location.shape)

    # process image
    start = time.time()

    # locations, feature_shapes = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    processingtime = time.time() - start
    print("processing time: ", processingtime)
   
    # Wurde hinzugefügt!
    # Verarbeitungszeit für jedes Bild wird in einer seperaten Textdatei in .\test gespeichert
    with open('test\\{}_time.txt'.format(image_fname), 'w') as timefile:
        timefile.write(str(processingtime))

    # correct for image scale
    boxes /= scale
    labels_to_locations = {}
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        
        # scores are sorted so we can break
        # Abänderung von 0.5 auf 0.45, da sonst keine Lokalisierungen stattfinden bei den trainierten Gewichtungen der Arbeit
        if score < 0.45:
            break
        start_x = int(box[0])
        start_y = int(box[1])
        end_x = int(box[2])
        end_y = int(box[3])
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
        # Zählen der Lokalisierungen pro Klasse
        if (label==0):
            apple_count += 1
        elif (label==1):
            kiwi_count += 1
        elif (label==2):
            coco_count += 1
        elif (label==3):
            tomato_count += 1

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', draw)

    # Wurde hinzugefügt!
    # Ausgabe Anzahl der Lokalisierungen
    print("Anzahl Äpfel: ", apple_count)
    print("Anzahl Kiwis: ", kiwi_count)
    print("Anzahl Kokosnüsse: ", coco_count)
    print("Anzahl Tomaten: ", tomato_count)

    # Wurde hinzugefügt!
    # Hier wird die Anzahl der Erkennungen jeder Obst- und Gemüsesorte jeweils für jedes Bild in eine seperate Textdatei gespeichert.
    with open('test\\{}_count.txt'.format(image_fname), 'w') as countfile:
        countfile.write("Anzahl Äpfel: " + str(apple_count) + '\n' +
                        'Anzahl Kiwis: ' + str(kiwi_count) + '\n' +
                        'Anzahl Kokosnüsse: ' + str(coco_count) + '\n' +
                        'Anzahl Tomaten: ' + str(tomato_count))

    key = cv2.waitKey(0)
    if int(key) == 121:
        image_fname = osp.split(image_path)[-1]
        cv2.imwrite('test/{}'.format(image_fname), draw)

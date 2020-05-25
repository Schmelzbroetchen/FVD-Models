"""
Dieser Code stammt ursprünglich von https://github.com/yunlongdong/EasyKerasYoloV3 und wurde für diese Arbeit leicht abgeändert.
Eigene Code-Abänderungen werden markiert.
"""

import sys
import argparse
from yolo import YOLO
from PIL import Image
import pdb
import glob
import os

def detect_img(yolo):
    """Lokalisiert Objekte auf den Testbildern im Test-Images-Pfad.
    Wurde abgeändert, sodass direkt auf allen Bildern im Ordner Test-Images lokalisiert wird."""
    count = 0
    for img in glob.glob(os.path.join("..\\..\\Datenbank\\Test-Images\\", "*.jpg")):
        count += 1
        #img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            r_image.save("results\\" + str(count) + ".jpg")
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path"),
		default='logs\\trained_weights_final.h5'
    )
    FLAGS = parser.parse_args()
    detect_img(YOLO(**vars(FLAGS)))

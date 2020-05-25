# YOLOv3 (Keras-Version)
Keras implementation of YOLOv3: You Only Live Once Version 3.
Cloned [from https://github.com/yunlongdong/EasyKerasYoloV3](https://github.com/yunlongdong/EasyKerasYoloV3)

## SETUP:

1. Neue Environment erstellen mit Python 3.6, um Requirements dort zu installieren (für diese Untersuchungen wurde Anaconda verwendet.)
2. Sicherstellen, dass TensorFlow Version 1.14.0 und Numpy installiert sind mit ```pip install tensorflow==1.14 --user``` und ```pip install numpy --user```
3. Requirements mit ```pip install -r requirements.txt --user``` installieren.

### Command zum Trainieren (für diese Arbeit):
```python train.py```

- Die Parameter für das Training (Pfade, etc.) werden in ```train.py``` festgelegt. Das Ausführen von ```train.py``` erstellt dann während des Trainings die Checkpoints und Logs im Ordner ```.\logs.```

- Für das Training sind die ```yolo_weights.h5``` nötig, die sich auf dem Stick in .\model_data befinden. Sie können aber auch unter [http://pjreddie.com/media/files/yolo.weights](http://pjreddie.com/media/files/yolo.weights) heruntergeladen werden.

### Command zum Testen (für diese Arbeit):
```python yolo_detect.py```

- Durch das Ausführen von ```yolo_detect.py``` werden die trainierten Gewichtungen auf alle Testbilder, die im Test-Images-Pfad zu finden sind, angewandt, wobei die Bilder mit eingezeichneten Begrenzungsrahmen, Scores und Klassifizierungen dann im ```.\results``` Ordner gespeichert werden.

## References
[1] [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) <br>
[2] [from https://github.com/yunlongdong/EasyKerasYoloV3](https://github.com/yunlongdong/EasyKerasYoloV3) <br>
[3] [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) <br>

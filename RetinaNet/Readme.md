# RetinaNet Keras-Version
Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.<br>
Cloned from [https://github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet).

## SETUP
1) Eine neue Environment mit Python 3.6 erstellen, in der die Requirements installiert werden sollen (dafür wurde Anaconda verwendet).
2) Mit ```pip install numpy --user``` Numpy installieren. Außerdem Tensorflow 1.15.0 installieren mit ```pip install tensorflow=1.14.0 --user```
4) Vom RetinaNet-Root-Ordner aus mit ```pip install . --user``` und sicherhaltshalber ```pip install -r requirements.txt --user``` die anderen Requirements installieren.
5) ```python setup.py build_ext --inplace``` ausführen.

- Es kann sein, dass die wichtigen .exe Dateien, wie retinanet-train.exe, retinanet-debug.exe, retinanet-evaluate.exe, retinanet-convert-model.exe nicht in der Bibliothek der Environment installiert werden. Falls der Fehler angezeigt wird, dass sie nicht im Path sind, das Python\Scripts Verzeichnis dem Path hinzufügen oder diese spezifischen .exe Dateien aus den Scripts des Python Ordners in die RetinaNet Environment verschieben (heißt ebenfalls Scripts).

### Command zum Trainieren:
```retinanet-train --imagenet-weights --batch-size 4 --steps 61 --epochs 30 --snapshot-path snapshots --tensorboard-dir logs csv ..\..\Datenbank\RetinaNet\retina_annotations_train.csv ..\..\Datenbank\RetinaNet\retina_classes_train.csv --val-annotations ..\..\Datenbank\RetinaNet\retina_annotations_test.csv```

- Dabei wird das RetinaNet mit den Default-Angaben (vortrainierte Resnet-Gewichtungen) auf den neuen Datensatz trainiert und es werden pro Epoche Checkpoints im ```.\logs``` Ordner erstellt.

### Vorgang Testen:
1) Das trainierte Modell wird zu einem Inference-Modell konvertiert mit dem Command: ```retinanet-convert-model snapshots\resnet50_csv_30.h5 snapshots\resnet50_csv_30_inference.h5``` (Auf dem Stick schon abgeschlossen)
2) Das Inference-Modell auf Real-Time-Bilder testen:
- Installieren von ```matplotlib```, falls es noch nicht vorhanden ist mit ```pip install matplotlib```.
- Im Ordner ```.\testing``` die Variablen ```model_path``` und ```img_name``` im Modul ```realtime-test.py``` anpassen und dann in ```.\testing``` direkt mit der Kommandozeile ausführen.

## References
[1] [https://github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)<br>
[2] [Focal Loss for Dense Object Detection (RetinaNet Paper)](https://arxiv.org/abs/1708.02002)

# FCOS Keras-Version
FCOS (Fully Convolutional One-Stage Object Detection) Implementation in Keras. <br>
Cloned from [https://github.com/xuannianz/keras-fcos](https://github.com/xuannianz/keras-fcos).

## SETUP

1) Environment für FCOS erstellen (in der Arbeit wurde Anaconda genutzt).
2) Sicherstellen, dass TensorFlow Version 1.15. installiert ist mit ```pip install tensorflow==1.14 --user```
3) Requirements installieren mit ```pip install -r requirements.txt --user```
4) ```python setup.py build_ext --inplace``` ausführen.

### Command zum Trainieren (für diese Arbeit):
```python train.py  --backbone resnet50 --batch-size 4 --epochs 200 --steps 61 --compute-val-loss --weighted-average csv ..\..\Datenbank\RetinaNet\retina_annotations_train.csv ..\..\Datenbank\RetinaNet\retina_classes_train.csv --val-annotations-path ..\..\Datenbank\RetinaNet\retina_annotations_test.csv```

- Das Ausführen von ```train.py``` erstellt dann die Logs im Ordner ```.\logs``` und die Checkpoints der trainierten Gewichtungen nach jeder Epoche jeweils in ```.\snapshots``` in einem seperat erstellten Ordner mit dem aktuellen Datum als Namen.

### Command zum Testen (für diese Arbeit):
```python inference.py```

- ```inference.py``` muss angepasst werden, falls andere Gewichtungen genutzt werden sollen. Das kann mit der Variable ```model_path``` verändert werden.
- In der GitHub-Version fehlen die trainierten Gewichte! Diese sind auf dem Stick verfügbar.
- Das Ausführen von ```inference.py``` testet alle Bilder im Test-Images Ordner und speichert die Ergebnisse mit eingezeichneten Begrenzungsrahmen, Scores und Klassifizierungen in ```.\test```. Es wird außerdem zu jedem Bild die Verarbeitungszeit, sowie die Anzahl der erkannten Obst- und Gemüsesorten in seperaten Textdateien jeweils gespeichert.

## References
[1] [https://github.com/tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)<br>
[2] [https://github.com/xuannianz/keras-fcos](https://github.com/xuannianz/keras-fcos)<br>
[3] [FCOS: Fully Convolutional One-Stage Object Detection (Paper)](https://arxiv.org/abs/1904.01355)

# Faster R-CNN (Keras-Version)
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.<br/>
cloned from [https://github.com/you359/Keras-FasterRCNN](https://github.com/you359/Keras-FasterRCNN)

## SETUP:

1) Eine neue Environment mit Python 3.6 erstellen, in der die Requirements installiert werden sollen (ich habe Anaconda dafür verwendet).
2) Mit 'pip install numpy --user' versichern, dass Numpy installiert ist.
3) Die anderen Requirements mit "pip install -r requirements.txt --user" installieren.

Die folgenden Commands müssen im Faster R-CNN Root-Ordner ausgeführt werden.

### Command zum Trainieren (für diese Arbeit):
python train_frcnn.py -o simple -p ..\Dataset-otherFormats\Faster-RCNN\train.txt --network resnet50 --num_epochs 200 --output_weight_path models\model_frcnn.hdf5 --input_weight_path .\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

- Das Ausführen von train_frcnn.py erstellt zu Beginn eine hdf5-Datei namens model_frcnn.hdf5, welche die Gewichtungen des trainierten Modells enthält und aktualisiert wird, sobald bessere Gewichtungen durch das Training entstehen. Ich habe aber auch noch eingebaut, dass nach der Aktualisierung des Checkpoints die Gewichtung jeder zweiten folgenden Epoche seperat abgespeichert werden, um variabel sehen zu können, ob es zwischendurch interessante Gewichtungen gab, welche man seperat testen könnte.

### Command zum Testen (für diese Arbeit):
python test_frcnn.py -p ..\..\Datenbank\Test-Images\ 

- Das Ausführen von test_frcnn.py erstellt in FasterR-CNN\results_imgs\ die Testbilder mit den eingezeichneten Lokalisierungen.

## Reference
[1] [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, 2015](https://arxiv.org/pdf/1506.01497.pdf) <br/>
[2] [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 2016](https://arxiv.org/pdf/1602.07261.pdf) <br/>
[3] [https://github.com/yhenon/keras-frcnn/](https://github.com/yhenon/keras-frcnn/)
[4] [https://github.com/you359/Keras-FasterRCNN](https://github.com/you359/Keras-FasterRCNN)

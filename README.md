# facial-recongnition
Live human-face recognition with OpenCV using Haar Cascade Classifiers; frontal_face dataset and LBPH Algorithm, works with small computing devices.

## How to run?
- Run ```face_add.py``` to genrate face samples, add id/name
- Train the sample images using ```face_train.py```
- Run facial recogniton script with ```facial_recognition.py```
- Control confidence/success level in ```face_recognition``` when face is detected using LBPH
- Edit/add id names in ```usr_id_label``` in facial_recognition.py

## Requirement
 - OpenCV, install using ```pip install opencv-contrib-python```

# References
- LBPH<a href="https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms"> docs</a></a> compressive guide. Sample code is in C++, you might be interested.

- <a href="https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html">Haar-cascade Detection</a>

- <a href="https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html">Cascade Classifier Training</a>

- Haarcascades pre-trained <a href="https://github.com/opencv/opencv/tree/master/data/haarcascades">dataset</a> useful for various subjects.

- Marcelo for introducing <a href="https://github.com/Mjrovai/OpenCV-Face-Recognition/blob/master/FacialRecognition/03_face_recognition.py"> confidences</a> level.

- Aswinth similar <a href="https://circuitdigest.com/microcontroller-projects/raspberry-pi-and-opencv-based-face-recognition-system">  guide</a> using Raspberry Pi.

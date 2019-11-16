"""
    File name: face_recognition.py
    Course Module: CU6051NA
    Author: Vijay Pathak
    Description: A basic face recognition program using OpenCV, Haar cascade, LBPH algorithm
    Date created: 30/12/2018
    Date last modified: 02/09/2019
    Python Version: 3.7.1
"""

# Import libraries
import cv2

# Create LBPH histograms for face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/data.yml')  # Read trained data

# Import face dataset
dataset_face = 'dataset/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(dataset_face)  # Create classifier

font = cv2.FONT_HERSHEY_SIMPLEX = 2  # Font

# Initialize user id
usr_id = 0

# User id label name
usr_id_label = ['user_0', 'user_1', 'user_2', 'user_3']

# Initialize and capture video frame
live_cam = cv2.VideoCapture(0)

print("\nPlease select the web-cam window and press 'ESC' key to exit.\n")

while True:

    ret, img = live_cam.read()  # Capture frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect face sizes
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10
    )

    for (x, y, w, h) in faces:

        end_crd_x = x + w  # face start and end coordinates
        end_crd_y = y + h

        cv2.rectangle(img, (x - 20, y - 20), (end_crd_x, end_crd_y), (0, 255, 33), 1)  # color, stroke

        usr_id, confidence = recognizer.predict(gray[y:end_crd_y, x:end_crd_x])

        # Check confidence, set confidence (100 subtract(-) detected confidence) to limit face recognition
        # Round confidence when user id match
        if (confidence < 100):
            usr_id = usr_id_label[usr_id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            usr_id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Description texts in live video frame
        cv2.putText(img, str(usr_id), (x - 28, y - 40), font, 1, (0, 255, 33), 1)
        cv2.putText(img, str(confidence), (x + 100, y - 35), font, 0.7, (200, 200, 43), 1)

    cv2.imshow('Camera', img)  # Display video frame in rectangle

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' key to exit video
    if k == 27:
        break

live_cam.release()  # Stop video frame capture
cv2.destroyAllWindows()  # Close all active windows

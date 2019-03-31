"""
    File name: face_train.py
    Course Module: CU6051NA
    Author: Vijay Pathak
    Description: A basic face recognition program using OpenCV, Haar cascade, LBPH algorithm
    Date created: 31/12/2018
    Date last modified: 02/09/2019
    Python Version: 3.7.1
"""

# Import libraries
import cv2
import numpy as np
from PIL import Image
import os


# Create directory if unavailable
def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Create directory
check_dir('train')

# Directory location
img_loc = 'images'

# Create LBPH histograms for face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_faces = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')  # Face dataset


# Function to get images, labels
def getImagesAndLabels(img_loc):
    img_locs = [os.path.join(img_loc, f) for f in os.listdir(img_loc)]  # Get all files from path

    # Initializae empty list
    face_smpls = []
    f_id = []

    for img_loc in img_locs:

        # PIL_img = Image.open(img_loc).convert('L') # Get images and convert to grayscale

        PIL_img = Image.open(img_loc)  # Get images
        img_numpy = np.array(PIL_img, 'uint8')  # Covert to numpy arrays
        # print(img_numpy)

        id = int(os.path.split(img_loc)[-1].split("_")[1])  # Get images id
        faces = dataset_faces.detectMultiScale(img_numpy)  # Get faces from dataset

        # Add images in face samples
        for (x, y, w, h) in faces:
            face_smpls.append(img_numpy[y:y + h, x:x + w])
            f_id.append(id)

    return face_smpls, f_id  # Get arrays


print("\n[Trainig Facial Expressions] Please wait....")

(faces, f_id) = getImagesAndLabels(img_loc)  # Faces and face_id
recognizer.train(faces, np.array(f_id))  # Train

# Save model in .yml file
recognizer.write('train/data.yml')

# Print the total number of faces trained
print("\n[{0} face expressions has been trained]\n".format(len(np.unique(f_id))))

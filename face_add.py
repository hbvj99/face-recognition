# Import libraries
import os
import cv2

# Create directory if unavailable
def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Capture Video
live_cam = cv2.VideoCapture(0)

# Import pre-trained dataset
dataset_face = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

# Each person numeric face id message
face_id = input('\nPlease enter user id (i.e. 2) and press "ENTER":\n')

print("\[Initializing Camera]\nPlease look at the camera and make different facial expressions.")

count = 0  # Image count

check_dir('images')  # Create directory

while (True):
    ret, img = live_cam.read()  # Capture frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = dataset_face.detectMultiScale(  # Detect face sizes
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(5, 5)
    )

    for (x, y, w, h) in faces:
        end_crd_x = x + w  # face start and end coordinates
        end_crd_y = y + h

        # Crop video frame in rectangle
        cv2.rectangle(img, (x, y), (end_crd_x, end_crd_y), (0, 255, 33), 1)  # color, stroke
        count += 1  # Increment image count

        # Save to folder
        cv2.imwrite("images/usr_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('Camera', img)  # Display frame, rectangle in faces

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50:  # Take 50 face images, stop frame capture
        break

print("\n[Facial Expressions Captured!]\nPlease select the web-cam window and press 'ESC' key to exit.")

live_cam.release()  # Stop video
cv2.destroyAllWindows()  # Close all windows

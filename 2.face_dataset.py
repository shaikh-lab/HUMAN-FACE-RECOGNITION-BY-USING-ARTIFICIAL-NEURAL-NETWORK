########################################################################################################################
###------------------------- HUMAN FACE RECOGNITION BY USING ARTIFICIAL NEURAL NETWORK ------------------------------###
########################################################################################################################
###-------------------------------------2.face_dataset.py------------------------------------------------------------###
########################################################################################################################
###------------------------------------------------------------------------------------------------------------------###
###------------------------------------------------------------------------------------------------------------------###
########################################################################################################################
import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))

cam = cv2.VideoCapture(0)
# set video width
cam.set(3, 640)
# set video height
cam.set(4, 480)

#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
#face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
face_id = input('\n enter user id end press <return> ==>  ')


print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

#start detect your face and take 30 pictures

while (True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the dataset folder
        cv2.imwrite("dataset/face_" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    # Take 50 face sample and stop video
    elif count >= 50:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



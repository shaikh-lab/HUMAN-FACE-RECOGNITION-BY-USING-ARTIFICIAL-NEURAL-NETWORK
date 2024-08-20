########################################################################################################################
###------------------------- HUMAN FACE RECOGNITION BY USING ARTIFICIAL NEURAL NETWORK ------------------------------###
########################################################################################################################
###-----------------------------------------4.face_recognition.py----------------------------------------------------###
########################################################################################################################
###------------------------------------------------------------------------------------------------------------------###
###------------------------------------------------------------------------------------------------------------------###
########################################################################################################################

import cv2
import os
import numpy as np


path = os.path.dirname(os.path.abspath('dataset'))

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainerdata.yml')
cascadePath = 'Cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter, the number of persons you want to include
id = [0, 1,2]

#key in names, start from the second place, leave first empty
name = ['unknown', 'r', 'shaikh', 't', ]

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)

# set video widht
cam.set(3, 640)

# set video height
cam.set(4, 480)

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )


    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])



        if confidence < 100:

            id = name[id]

            confidence = "  {0}%".format(round(100 - confidence))

        else:

            id = "unknown"

            confidence = "  {0}%".format(round(100 - confidence))

        
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)

        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 4)
        #cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 255), 3)
    
    cv2.imshow('camera', img)


    k = cv2.waitKey(10) & 0xff   # Press 'ESC' for exiting video
    if k == 27:
        break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

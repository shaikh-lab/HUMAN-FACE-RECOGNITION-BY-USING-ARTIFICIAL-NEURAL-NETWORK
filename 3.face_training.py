########################################################################################################################
###------------------------- HUMAN FACE RECOGNITION BY USING ARTIFICIAL NEURAL NETWORK ------------------------------###
########################################################################################################################
###------------------------------------3.face_training.py------------------------------------------------------------###
########################################################################################################################
###------------------------------------------------------------------------------------------------------------------###
###------------------------------------------------------------------------------------------------------------------###
########################################################################################################################

import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = os.path.dirname(os.path.abspath('dataset'))
#path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath ="Cascades/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascadePath)



# function to get the images and label data

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # convert it to grayscale

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = getImagesAndLabels('dataset')

recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainerdata.yml

recognizer.write('trainer/trainerdata.yml')

# Print the numer of faces trained and end program

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

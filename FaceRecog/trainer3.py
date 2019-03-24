import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = 'Train/gav/'


def getImagesWithID(path):
    imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg= Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg, 'uint8')
        id = int(os.path.split(imagePath)[-1].split()[0])
        faces.append(faceNp)
        print(id)
        IDs.append(id)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, faces


IDs, faces = getImagesWithID(path)
recognizer.train(faces, np.array(IDs))
recognizer.save('Recognizer/Data1.yml')
cv2.destroyAllWindows()


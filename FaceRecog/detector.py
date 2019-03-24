import cv2
import numpy as np


faceCascade = cv2.CascadeClassifier('E:/FaceRecog/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('E:/FaceRecog/Recognizer/Data1.yml')
#id = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 4)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        print(id)
        if id==1 or id==2:
            id="Unlock"
        else:
            id="Locked"
        cv2.putText(img, str(id), (x+w, y+h), font, fontScale, fontColor, 3)
    # Display the resulting frame
    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
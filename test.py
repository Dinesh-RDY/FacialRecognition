import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
model = keras.models.load_model("model1ANN.h5")  # type: ignore
labels = []
for dir in os.listdir("data/"):
    labels.append(dir)
print(model.summary())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
while True:
    _ , img = cam.read()
    img = cv2.flip(img, 1)
    faces = detector.detectMultiScale(img)
    for (x , y , w , h) in faces:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray[y- 30:y+h +30, x - 30:x+w + 30]
            gray = cv2.resize(gray , (256 ,256))
        except:
            continue
        cv2.imshow("gray" , gray)
        gray = gray.reshape(-1, 256, 256, 1).astype('float32') / 255.
        ans = model.predict(gray)
        print(ans)
        n = np.argmax(ans)
        try:
            cv2.rectangle(img , (x , y) , ( x+ h , y + w ) , (0,255,0) , 2)
            cv2.putText(img ,  str(labels[n]), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        except:
            continue
        
    cv2.imshow("image" ,img)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
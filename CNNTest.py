import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
model = keras.models.load_model("modelCNN.h5")  # type: ignore
labels = []
for dir in os.listdir("dataCNN/"):
    labels.append(dir)
print(model.summary())
cam = cv2.VideoCapture(0)
while True:
    _ , img = cam.read()
    # img = cv2.flip(img, 1)
    cv2.imshow("image" , img)
    img = cv2.resize(img, (256,256))
    img = np.array(img)
    # img = np.reshape(img, (256,256,3))
    # img = img / 255
    cv2.imshow("image" ,img)
    img = img.reshape(-1, 256, 256, 3) / 255 #type:ignore
    ans = model.predict(img)
    print(ans)
    n = np.argmax(ans)
    print(labels[n])
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
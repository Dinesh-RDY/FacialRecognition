import cv2
import os

cam = cv2.VideoCapture(0)
face_id = input("Enter roll number")
count = 0
os.mkdir(os.path.join(os.getcwd() , "dataCNN/" + face_id))
while (True):
    ret, img = cam.read()
    # img = cv2.flip(img, 1)
    cv2.imshow('image', img)
    img = cv2.resize(img , (256 , 256))
    cv2.imwrite(f"dataCNN/{face_id}/" + str(count) + ".jpg", img)
    count += 1
    cv2.imshow('image', img)
    k = cv2.waitKey(10) & 0xff  
    if k == ord('q'):
        break
    elif count > 200:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

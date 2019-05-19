import cv2
import numpy as np
import sys
import dropbox
import os
from pushover import init, Client
import time

init("aypdp8ionpppkenbx8w3irxhf6e855")

#dropbox unique token
token = 'Bj5eYdrWuUAAAAAAAAAAEtmDseR8WsNoxII72K_7z7n-w0f8aBprpJUrkrES3Rr9'
dbx = dropbox.Dropbox(token)

#check api, kust for testing purpose
#print(dbx.users_get_current_account())


facePath = "haarcascade_frontalface_default.xml"
faceClassifier = cv2.CascadeClassifier(facePath)

img_cnt = 0

cap = cv2.VideoCapture(0)
cap.set(3,480) #set width
cap.set(4,480) #set height

def uploadImg(file):
    path = file
    dbx.files_upload(file, path)
    

while True:
    ret, frame = cap.read()
    image = frame
    
    if not frame.any():
        break
        
    color = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(
        color,
        scaleFactor = 1.05,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
        )


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, x+h), (255, 0, 0), 2)
        
    cv2.imshow('Face', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #QUIT ON Q PRESS
        break
    
    if cv2.waitKey(1) & 0xFF == ord('s'): #save image on s press
        img_name = "/home/pi/Desktop/Images/opencv_frame_{}.png".format(img_cnt)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        uploadImg(img_name)
        img_cnt += 1
        Client("uyaqqdmp4cfv1eqbmckqfoyvudyjc1").send_message("Image uploaded on Dropbox", title = "New upload")

cap.release()
cv2.destroyAllWindows()
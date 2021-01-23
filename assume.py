import cv2
import pygame
from pygame import mixer
import time


count=0#variable pour faire la décision
#utilisation de classificateurs

eye_cascPath = 'haar cascade files\haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'haar cascade files\haarcascade_frontalface_alt.xml'  #face detect model
mouth ='haar cascade files\Mouth.xml' #mouth detect model

mouth_cascade = cv2.CascadeClassifier(mouth)
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

#cette fonction declanche une alarme
def alert():
    mixer.init()
    alert = mixer.Sound('alarm.wav')
    alert.play()
    time.sleep(1)


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()#ouverture de frame
    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
        # Draw a rectangle around the faces
            for (x, y, w, h) in faces:

             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
             #encadrer l'image capturé
             # boxing face cropped
             frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
             frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
             roi_color = frame[y:y + h, x:x + w]
             eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
        #detectMultiScale() [1/3] Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles. Matrix of the type CV_8U containing an image where objects are detected

            mouth = mouth_cascade.detectMultiScale(frame, 1.3, 5)
            # Draw a rectangle around the faces
            for (x, y, w, h) in mouth:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #ici si les yeux sont fermés et le nbre de print wakeup>18 il declanche l'alarme, en effet la variable count joue le role de chronometre
            if len(eyes) == 0 :
                if len(mouth) == 0:

                  count+=1
                  print('please,wake up!!')
                  if (count>18):
                      alert()




            else:
                mixer.init()
                pygame.mixer.music.stop()
                print('very good ,if you feel tired , stop a while!')



        frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Face Recognition',frame_tmp)

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            break
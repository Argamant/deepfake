import cv2
import re
import os
import ntpath
# Load the cascade
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def crop_face(imagePath):
    img = cv2.imread(imagePath)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize= (60,60))
    # Draw rectangle around the faces
    if len(faces) > 0:
        firstFace = faces[0]
        (x, y, w, h) = firstFace
        gray_eye = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_eye, scaleFactor = 1.1, minNeighbors = 5)
        if len(gray_eye) > 0:
          crop_img = img[y:y+h, x:x+w]
          return crop_img
        else:
          return None
    else:
       return None

def crop_face_save_jpg(imagePath):
    img = crop_face(imagePath)
    if img is None:
      print(imagePath, " did not have a face")
      os.remove(imagePath)
    else:
      #faceImagePath = re.sub(r'(.*?).jpg$', r"\1_face.jpg", imagePath, flags=re.DOTALL)
      #cv2.imwrite(imagePath, img)
      #fileName = ntpath.basename(imagePath)
      #faceImagePath = 'C:\\Users\\rugge\\source\\repos\\deepfake\\deepfake\\demo\\faces\\' + 'face_' + fileName
      base_path=re.sub(r'(.jpg$)', '', imagePath)
      faceImagePath = base_path +  '_face.jpg'
      cv2.imwrite(faceImagePath, img)
      os.remove(imagePath)

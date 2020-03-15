import cv2
import re
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def crop_face(imagePath):
    img = cv2.imread(imagePath)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    if len(faces) > 0:
        firstFace = faces[0]
        (x, y, w, h) = firstFace
        crop_img = img[y:y+h, x:x+w]
        return crop_img
    else:
       return None

def crop_face_save_jpg(imagePath):
    img = crop_face(imagePath)
    if img is None:
      print(imagePath, " did not have a face")
      os.remove(imagePath)
    else:
      faceImagePath = re.sub(r'(.*?).jpg$', r"\1_face.jpg", imagePath, flags=re.DOTALL)
      cv2.imwrite(imagePath, img)

crop_face_save_jpg("test.jpg")

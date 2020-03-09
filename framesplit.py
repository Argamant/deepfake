#basic method for separating a video into frames
from os import mkdir
import cv2
vidcap = cv2.VideoCapture('big_and_bootiful.mp4')
mkdir('big_and_bootiful/')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("big_and_bootiful/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
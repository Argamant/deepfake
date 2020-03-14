#basic method for separating a video into frames
from os import mkdir
import cv2
from json import load
from os import system
from glob import glob

def extract(source_dir, destination, delim='/')
  file_list = glob(source_dir + '*.mp4')

  labels = open(source_dir + 'metadata.json', "r")
  data = load(labels)
  os.mkdir(destination + 'frames' + delim)
  for vid in file_list:
      #the file name without the .mp4 file extension
      key = vid.split(delim)[-1][:-4]
      system('ffmpeg -i {} -vf \"scale=320:240,fps=1\" \"'.format(vid) + destination + 'frames\\{}_%04d.jpeg\"'.format(key))
      #system('ffmpeg -i {} \"'.format(vid) + destination + 'audio\\{}.wav\"'.format(key))
      i+=1
      
  labels.close()
  
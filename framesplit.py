#basic method for separating a video into frames
from os import mkdir
import cv2
from json import load
from os import system
from glob import glob
import random

def extract(source_dir, destination, test_split = 0.2, delim='/'):
  file_list = glob(source_dir + '*.mp4')
  random.shuffle(file_list)
  labels = open(source_dir + 'metadata.json', "r")
  data = load(labels)
  #os.mkdir(destination + 'frames' + delim)
  #key = vid.split(delim)[-1][:-4]
  #print(data[file_list[0][-1][:-4]])
  paths = [destination + delim + 'test' + delim, destination + delim + 'train' + delim, destination + delim + 'validation' + delim]
  try:
    mkdir(paths[0])
    mkdir(paths[0] + 'real' + delim)
    mkdir(paths[0] + 'fake' + delim)

    mkdir(paths[1])
    mkdir(paths[1] + 'real' + delim)
    mkdir(paths[1] + 'fake' + delim)

    mkdir(paths[2])
    mkdir(paths[2] + 'real' + delim)
    mkdir(paths[2] + 'fake' + delim)
  except:
    pass
  test_split = int(len(file_list)*test_split)
  for key in data[:test_split]:
      #the file name without the .mp4 file extension
        #key = vid.split(delim)[-1][:-4]
        #real_fake = data[vid.split(delim)[-1]]['label']=='REAL'
        real_fake = data[key]['label']=='REAL'
        if real_fake:
            full_dest =  paths[0] + 'real' + delim
        else:
            full_dest =  paths[0] + 'fake' + delim

        #system('ffmpeg -i {} -vf \"scale=320:240,fps=1\" \"'.format(vid) + full_dest + '{}_%08d.jpeg\"'.format(key))
      
      # system('ffmpeg -i {} \"'.format(vid) + destination + 'audio'+delim+'{}.wav\"'.format(key))
      #i+=1
      
  labels.close()


extract('G:\\deepfake\\dfdc_train_part_04\\dfdc_train_part_4\\', 'G:\\deepfake\\data\\', delim = '\\')
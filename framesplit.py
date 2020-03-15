#basic method for separating a video into frames
from os import mkdir
import cv2
from json import load
from os import system
from glob import glob

def extract(source_dir, destination, delim='/'):
  file_list = glob(source_dir + '*.mp4')

  labels = open(source_dir + 'metadata.json', "r")
  data = load(labels)
  #os.mkdir(destination + 'frames' + delim)
  #key = vid.split(delim)[-1][:-4]
  #print(data[file_list[0][-1][:-4]])

  for vid in file_list:
      #the file name without the .mp4 file extension
        key = vid.split(delim)[-1][:-4]
        real_fake = data[vid.split(delim)[-1]]['label']=='REAL'
        if real_fake:
            full_dest =  destination + 'real' + delim
        else:
            full_dest =  destination + 'fake' + delim

        system('ffmpeg -i {} -vf \"scale=320:240,fps=1\" \"'.format(vid) + full_dest + '{}_%08d.jpeg\"'.format(key))
      
      # system('ffmpeg -i {} \"'.format(vid) + destination + 'audio'+delim+'{}.wav\"'.format(key))
      #i+=1
      
  labels.close()


extract('C:\\Users\\rugge\\source\\repos\\deepfake\\deepfake\\data\\dfdc_train_part_02\\dfdc_train_part_2\\', 'G:\\deepfake\\dfdc_train_part_2\\', '\\')
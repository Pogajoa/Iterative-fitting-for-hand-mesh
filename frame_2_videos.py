import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('./concatenate/results/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
FPS = 90
fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
out = cv2.VideoWriter('test.mp4', fourcc, 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
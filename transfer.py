import cv2
import numpy as np
import os
prov_list = os.listdir('./concatenate/images/provided')
prov_list.sort(key=lambda x:int(x.split('.')[0]))
#print(prov_list)
for name in prov_list:
    img1 = cv2.imread("./youtube_3d_hands/data/youtube/G26Ly2CTUy0/video/frames/" + name)
    cv2.imwrite('./concatenate/images/original/' + name, img1)    

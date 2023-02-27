import cv2
import numpy as np
import os

img_list1 = os.listdir('./concatenate/images/original')
img_list2 = os.listdir('./concatenate/images/provided')
img_list3 = os.listdir('./concatenate/images/iterated')

img_list1.sort(key = lambda x:int(x.split('.')[0]))
img_list2.sort(key = lambda x:int(x.split('.')[0]))
img_list3.sort(key = lambda x:int(x))
for i in range(len(img_list1)):
    img1 = cv2.imread("./concatenate/images/original/" + img_list1[i])
    img2 = cv2.imread("./concatenate/images/provided/" + img_list2[i])
    img3 = cv2.imread("./concatenate/images/iterated/" + img_list3[i] + '/000/output.png')

    img1 = cv2.resize(img1, (450, 400))
    img2 = cv2.resize(img2, (450, 400))
    img3 = cv2.resize(img3, (450, 400))

    #img_con = np.concatenate([img, img2], axis=0)
    img_con = np.concatenate([img1, img2, img3], axis=1)

    save_file="./concatenate/results/" + str(i) + ".png"
    cv2.imshow('show', img_con)
    cv2.imwrite(save_file, img_con) 
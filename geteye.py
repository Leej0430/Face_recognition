import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 
import dlib

#display image
img = mpimg.imread('이준엽1.jpg')
# imgplot = plt.imshow(img)
# plt.show()

eyedetect = cv2.CascadeClassifier('haarcascade_eye.xml')
geteye = eyedetect.detectMultiScale(img)

gozler =[]
for(x,y,w,h) in geteye:
    gozler.append(img[y:y+h,x:x+w])
imgplot= plt.imshow(gozler[0])
plt.show()
imgplot= plt.imshow(gozler[1])
plt.show()
imgplot= plt.imshow(gozler[2])
plt.show()
imgplot= plt.imshow(gozler[3])
plt.show()




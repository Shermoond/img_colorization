

from keras.utils import load_img,img_to_array
from keras.layers import Conv2D, UpSampling2D
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage import color
import numpy as np
import os
import tensorflow as tf
import keras
import cv2 as cv2
import pandas

pathG="C:\\Users\\lstat\\source\\repos\\Personal Assistant\\img_colorization\\image\\"
imagesG=[]
for img in os.listdir(pathG):
  img=pathG+img
  img=load_img(img,target_size=(100,100))
  img = color.rgb2gray(img)
  img=img_to_array(img)/ 255

  
  imagesG.append(img)
pathC="C:\\Users\\lstat\\source\\repos\\Personal Assistant\\img_colorization\color\\"
imagesC=[]
for img in os.listdir(pathC):
  img=pathC+img
  img=load_img(img,target_size=(100,100))
  img=img_to_array(img)/255
  labimg=rgb2lab(img)
  labnormed=(labimg+[0,128,128]) / [100,255,255]
  Y=labnormed[:,:,1:]
  imagesC.append(Y)
X=np.array(imagesG)
Y=np.array(imagesC)

#Input Layer
x1=keras.Input(shape=(None,None,1))
x2=Conv2D(8,(3,3),activation="relu",padding="same",strides=2)(x1)
x3=Conv2D(16,(3,3),activation="relu",padding="same")(x2)
x4=Conv2D(16,(3,3),activation="relu",padding="same",strides=2)(x3)
x5=Conv2D(32,(3,3),activation="relu",padding="same")(x4)
x6=Conv2D(32,(3,3),activation="relu",padding="same",strides=2)(x5)
#upsampling layer
x7=UpSampling2D((2,2))(x6)
x8=Conv2D(32,(3,3),activation="relu",padding="same")(x7)
x9=UpSampling2D((2,2))(x8)
x10=Conv2D(16,(3,3),activation="relu",padding="same")(x9)
x11=UpSampling2D((2,2))(x10)
x12=Conv2D(2,(3,3),activation="sigmoid",padding="same")(x11)

x12=tf.reshape(x12,(104,104,2))
x12=tf.image.resize(x12,[100,100])
x12=tf.reshape(x12,(1,100,100,2))

model=keras.Model(x1,x12)
counter=0
for layer in model.layers:
  if (layer._name=="conv2d"):
      layer._name = layer.name + str(counter)
      counter+=1
for inputs in model.inputs:
  print(inputs._name)
model.compile(optimizer='rmsprop',loss="mse")
model.fit(X,Y,batch_size=1,epochs=10,verbose=1)
model.evaluate(X,Y,batch_size=1)
img="im237.jpg"
img="C:\\Users\\lstat\\source\\repos\\Personal Assistant\\img_colorization\\color\\im1.jpg"
img2=load_img(img,target_size=(100,100))
img2=img_to_array(img2)/255
ss=img2.shape
img=load_img(img,target_size=(100,100))
img=color.rgb2gray(img)
img=img_to_array(img)/255
imshow(img)
plt.show()
print(img.shape)
x=np.array(img)
print(x.shape)
x=np.expand_dims(x,axis=2)

x=np.reshape(x,(-1,100,100,1))



output=model.predict(x)
print(output)
output=np.reshape(output,(100,100,2))
output=cv2.resize(output,(ss[0],ss[1]))




ABimg=output

outputLAB=np.zeros((ss[0],ss[1],3))
outputLAB[:,:,0]=img2[:,:,0]
outputLAB[:,:,1:]=ABimg[:,:,:]
outputLAB=(outputLAB*[100,255,255]-[0,128,128])
print(outputLAB)
rgbimg=lab2rgb(outputLAB)
imshow(outputLAB)
plt.show()
print(rgbimg)
imshow(rgbimg)
plt.show()




from skimage import color
from skimage import io
from skimage import util
import os

pathC="C:\\Users\\llabr\\Desktop\\color\\"
imagesC=[]
counter=1
for img in os.listdir(pathC):
    img=pathC+img
    img = io.imread(img)
    img =util.img_as_uint(img)
    img=color.rgb2gray(img)
    io.imsave("C:\\Users\\llabr\\Desktop\\gray\\im"+str(counter)+".jpg",img)
    counter=counter+1



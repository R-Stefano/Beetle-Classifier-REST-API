import numpy as np
import time
import cv2
import glob
import keras
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

#sometimes 20%
sometimes_2= lambda aug: iaa.Sometimes(0.2, aug)
#sometimes 50%
sometimes_5= lambda aug: iaa.Sometimes(0.5, aug)
sometimes_75= lambda aug: iaa.Sometimes(0.75, aug)


#Apply between 0 and 2 of the following augmentations
seq=iaa.SomeOf((1,3),
                [
                    iaa.Fliplr(0.5),#horizontally flip 20% of the images
                    iaa.Flipud(0.5),#vertically flip the images
                    sometimes_75(iaa.CropAndPad(percent=(-0.05, 0.1),pad_mode=ia.ALL,pad_cval=(0, 255))), #20% of the images are crop the image by -5% and 10%
                    sometimes_75(iaa.Scale({"height": (0.2, 0.75), "width": (0.2, 0.75)})), #20% of the images are scale by 50-75% of its original size
                    sometimes_75(iaa.AdditiveGaussianNoise(0, 0.15*255))
                ], random_order=True)


def loadPrepareImages(path,imgsize):
  imgList = []
  Labels = []
  Folders = []

  files = glob.glob (path)

  tmpfolder = ""

  i = -1
  for myFile in files:   
      image = cv2.imread(myFile)
      imgList.append(image)

      #find where is the second /
      folderlength = myFile.find('/',10,-1)
      #obtain the foldername cropping the file path string
      folderName = myFile[len("images/"):folderlength] 

      if tmpfolder != folderName:
        i+=1
        tmpfolder = folderName
        Folders.append(tmpfolder)
      Labels.append(i)
  
  #Store the unique images into a single matrix of shape [images,256,256,3]
  numberUniqueImages=len(imgList)
  X=np.zeros((numberUniqueImages,imgsize,imgsize,3))

  for idx, value in enumerate(imgList):
    X[idx] = cv2.resize(value, (imgsize,imgsize))
  
  #Convert list of lables into a numpy vector
  Y=np.asarray(Labels)
  
  x_train,x_test, y_train, y_test=train_test_split(X,Y,test_size=0.1)
  
  return x_train,x_test, y_train, y_test
  
def augmentData(batch):
  out=np.zeros((batch.shape))
  images_aug = seq.augment_images(np.uint8(batch))
    
  for idx,el in enumerate(images_aug):
    out[idx]=cv2.resize(el, (256,256))
       
  return out/255.
import numpy as np
import cv2
import glob
import os
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
import tensorflow as tf

seq = iaa.SomeOf((0,3),[
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),#vertically flip the images
    iaa.Crop(percent=(0, 0.2)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-360, 360),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

def loadData(path,imgsize, classes):
  X = []
  Y = []
  Folders = []

  num_class=0
  for folder in os.listdir(path):
    if(num_class<classes):
      for img in os.listdir(path+"/"+folder):
        image = cv2.imread(path+"/"+folder+"/"+img)
        image=cv2.resize(image, (imgsize,imgsize))
        X.append(image)
        Y.append(num_class)
      
      num_class+=1

  #Convert lists into a numpy
  X=np.asarray(X)
  Y=np.asarray(Y)

  x_train,x_test, y_train, y_test=train_test_split(X,Y,test_size=0.15)

  return x_train,x_test, y_train, y_test

def augmentData(batchIn,imgsize):
  out=np.zeros((batchIn.shape))
  images_aug = seq.augment_images(np.uint8(batchIn))

  for idx, img in enumerate(images_aug):
    out[idx]=cv2.resize(img, (imgsize,imgsize)) 
  return out/255.
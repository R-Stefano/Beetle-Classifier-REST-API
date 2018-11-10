import numpy as np
import cv2
import glob
import os
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
import tensorflow as tf
#sometimes 20%
sometimes_2= lambda aug: iaa.Sometimes(0.2, aug)
#sometimes 50%
sometimes_5= lambda aug: iaa.Sometimes(0.5, aug)
sometimes_75= lambda aug: iaa.Sometimes(0.75, aug)

'''
convolveMatrix = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])
#Apply between 0 and 2 of the following augmentations
seq=iaa.SomeOf((0,2),
                [
                    iaa.Fliplr(0.5),#horizontally flip 50% of the images
                    iaa.Flipud(0.5),#vertically flip the images
                    iaa.CropAndPad(percent=(-0.05, 0.1),pad_mode=ia.ALL,pad_cval=(0, 255)), #20% of the images are crop the image by -5% and 10%
                    iaa.Scale({"height": (0.2, 0.75), "width": (0.2, 0.75)}), #20% of the images are scale by 50-75% of its original size
                    iaa.Affine(rotate=(0,360)),
                    iaa.OneOf([
                      iaa.WithChannels((0,2), iaa.Add((10, 100))),
                      iaa.AdditiveGaussianNoise(0, 0.15*255),
                      iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128)),
                      iaa.GaussianBlur(sigma=(0.0, 3.0)),
                      iaa.AverageBlur(k=(2, 11)),
                      iaa.MedianBlur(k=(3, 11)),
                      iaa.AverageBlur(k=((5, 11), (1, 3))),
                      iaa.EdgeDetect(alpha=(0.0, 1.0)),
                      iaa.Invert(0.25, per_channel=0.5)
                      iaa.Add((-40, 40), per_channel=0.5),
                      iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                      iaa.Multiply((0.5, 1.5)),
                    ]),
                    iaa.Dropout(p=(0, 0.2)),
                    iaa.CoarseDropout(0.02, size_percent=0.5),
                ], random_order=True)
'''

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
  
  newX=X.copy()

  x_train,x_test, y_train, y_test=train_test_split(X,Y,test_size=0.15)

  X=x_train.copy()/255.
  Y=y_train.copy()
  for i in range(5):
    aug=augmentData(x_train, imgsize)
    X=np.concatenate((X, aug), axis=0)
    Y=np.concatenate((Y, y_train))

  return X,x_test, Y, y_test

def augmentData(batch,imgsize):
  out=np.zeros((batch.shape))
  images_aug = seq.augment_images(np.uint8(batch))
    
  for idx,el in enumerate(images_aug):
    out[idx]=cv2.resize(el, (imgsize,imgsize))       
  return out/255.
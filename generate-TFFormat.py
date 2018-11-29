import numpy as np
import os
import cv2
import tensorflow as tf
from object_detection.utils import dataset_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

def create_tf_example(filename, image, className, classIdx, imgsize):
    height = imgsize # Image height
    width = imgsize # Image width
    encoded_image_data = image # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        #'image/height': dataset_util.int64_feature(height),
        #'image/width': dataset_util.int64_feature(width),
        #'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        #'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/class/text': dataset_util.bytes_feature(className.encode('utf-8')),
        'image/class/label': dataset_util.int64_feature(int(classIdx)),
    }))
    
    return tf_example

imgsize=350
dataset_folder="dataset"

#Used to define the number of train examples and test examples
tot_count=0
#Every folder inside dataset is a class
for className in (os.listdir(dataset_folder)):
    count=0
    #iterate through the images inside the folder(class)
    for img in (os.listdir(dataset_folder+"/"+className)):
        count+=1
    print("Class name: " + className + " Number images: "+str(count+1))
    tot_count+=count

print("Total images: {:.0f} Train images: {:.0f} test images: {:.0f}".format(tot_count, tot_count*0.8, tot_count*0.2))

num_train_examples=int(tot_count)*0.8
num_test_examples=int(tot_count)*0.2


#Prepare the files for store TFRecords
shard_train="data/train-dataset.record"
shard_test="data/test-dataset.record"
test_count=0
train_count=0

test_num_shards=10
train_num_shards=test_num_shards*4 #keep 80% train data, 20% test data

examples_shard=int(num_train_examples//train_num_shards+1)

with contextlib2.ExitStack() as tf_record_close_stack:
    train_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, shard_train, train_num_shards)
    
    test_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, shard_test, test_num_shards)

    classIdx=-1
    #Retrieve the folders 
    for className in (os.listdir(dataset_folder)):
        #Every folder is a class, so define the class name
        classIdx +=1

        #iterate through the images inside the folder
        for img in (os.listdir(dataset_folder+"/"+className)):
            image=cv2.imread(dataset_folder+"/"+className+"/"+img)
            #cv2 read as BGR, convert to rgb
            img=image[:,:,::-1]

            #resize image
            #img=cv2.resize(rgbImg, (imgsize, imgsize))

            #encode it
            encodedImg=cv2.imencode('.png', img)[1].tostring()

            if (np.random.rand()<0.2 and test_count<test_num_shards*examples_shard):
                imageName="img_"+str(test_count)+".png"
                tf_example = create_tf_example(imageName, encodedImg, className, classIdx, imgsize)

                output_shard_index = test_count //examples_shard
                test_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                
                test_count+=1
            elif(train_count<train_num_shards*examples_shard):
                imageName="img_"+str(train_count)+".png"
                tf_example = create_tf_example(imageName, encodedImg, className, classIdx, imgsize)

                output_shard_index = train_count //examples_shard
                train_tfrecords[output_shard_index].write(tf_example.SerializeToString())

                train_count+=1
    
    print("Done, train size: {:.0f}   test size: {:.0f}".format(train_count, test_count))
        
        
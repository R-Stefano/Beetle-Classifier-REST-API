import numpy as np
import time
import tensorflow as tf
from model import Model
import imagePreprocesser

learn_rate=0.00001
imgsize=256
epoches=1000
batch_size=125

path="images/*/*.jpg"
x_train, x_test, y_train, y_test=imagePreprocesser.loadPrepareImages(path,imgsize)

#need the variable batch_size, classes
trainData=x_train.shape[0]
classes=36


global_step=0
tf.reset_default_graph()
with tf.Session() as sess:
  model=Model(learn_rate, imgsize,classes)
  
  sess.run(tf.global_variables_initializer())
  sess.run(model.init_acc_metrics)

  file=tf.summary.FileWriter('tensorboard/', sess.graph)
  saver=tf.train.Saver()
  for epoch in range(epoches):
    start = time.time()
    
    #compute the accuracy
    print(">>Computing accuracy..")
    _, summ=sess.run([model.acc, model.accSummary], feed_dict={model.input: x_test, model.labels: y_test})
    file.add_summary(summ, global_step)
    

    print(">>Training model..")
    for startB in range(0, trainData, batch_size):
      endB=startB+batch_size

      batchInput=x_train[startB:endB]
      batchLabels=y_train[startB:endB]

      #Data augmentation
      batchInput=imagePreprocesser.augmentData(batchInput)

      _, summ=sess.run([model.opt, model.stepSummary], feed_dict={model.input: batchInput, model.labels: batchLabels})

      file.add_summary(summ, global_step)
      global_step+=1
      
    if epoch%25==0:
      saver.save(sess, "agentBackup/graph.ckpt")
      
    end = time.time()
    elapsed = end - start
    print("Epoch in {:.4f}".format(elapsed))
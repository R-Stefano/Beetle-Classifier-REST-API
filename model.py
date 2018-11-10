import tensorflow as tf
layer=tf.contrib.layers

class Model():
  def __init__(self, learning_rate, imgsize, classes):
    self.lr=learning_rate
    self.imgsize=imgsize
    self.num_classes=classes
    
    self.buildNetwork()
    self.buildTraining()
    self.buildTestAccuracy()
    
    self.buildTensorboard()
   
  def convolution(self,x, num_out, kernel, stride):
    x=layer.conv2d(x, num_out, kernel, stride,
                   normalizer_fn=tf.layers.batch_normalization,
                   activation_fn=tf.nn.leaky_relu)
    return x
   
  
  def buildNetwork(self):
    self.input=tf.placeholder(tf.float32, shape=[None, self.imgsize, self.imgsize, 3], name="input_image")
    
    x=self.convolution(self.input, 32, 3, 1)
    x=self.convolution(x, 64, 3, 2)
    x=self.convolution(x, 128, 3, 2)
    x=self.convolution(x, 256, 3, 2)
    x=self.convolution(x, 512, 3, 2)
   
    shape=x.get_shape().as_list()
    flattenVec=tf.reshape(x, (-1, shape[1]*shape[2]*shape[3]))
    
    dropout=tf.layers.dropout(flattenVec, rate=0.5)
    x=tf.layers.dense(dropout, 512, activation=tf.nn.leaky_relu)
    self.logits=tf.layers.dense(x, self.num_classes, activation=None)
  
  def buildTraining(self):
    self.labels=tf.placeholder(tf.int32, shape=[None], name="labels")
    self.hotEncoded=tf.one_hot(indices=self.labels ,depth=self.num_classes)
    
    errors=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.hotEncoded, logits=self.logits)
    
    self.meanE=tf.reduce_mean(errors)
    
    optimizer=tf.train.AdamOptimizer(self.lr)
    self.opt=optimizer.minimize(self.meanE)
  
  def buildTestAccuracy(self):
    self.acc, self.t=tf.metrics.accuracy(labels=tf.math.argmax(self.hotEncoded, axis=1), predictions=tf.math.argmax(self.logits, axis=1), name="accuracy")   
    
    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")

    # Define initializer to initialize/reset running variables
    self.init_acc_metrics= tf.variables_initializer(var_list=running_vars)
  
  def buildTensorboard(self):
    self.stepSummary=tf.summary.merge([tf.summary.scalar("loss", self.meanE)])
    self.accSummary=tf.summary.merge([tf.summary.scalar("accuracy", self.acc)])
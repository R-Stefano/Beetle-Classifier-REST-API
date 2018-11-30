from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
slim = tf.contrib.slim   

tf.app.flags.DEFINE_string(
    'checkpoint_path', '', 'The name of the architecture to save.')

tf.app.flags.DEFINE_string(
    'step', '', 'The ckpt- number that you want to use.')

tf.app.flags.DEFINE_string(
    'output_file', 'exportedModel.pb', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS
def main(_):
    if not FLAGS.checkpoint_path:
        raise ValueError('You must supply the path to load the model with --checkpoint_path')
    
    if not FLAGS.step:
        raise ValueError('You must supply the step from which checkpoint load the model with --step')

    from google.protobuf import text_format
    gf = tf.GraphDef()
    text_format.Merge(open(FLAGS.checkpoint_path+'graph.pbtxt','rb').read(), gf)
    tensors=[n.name for n in gf.node if n.op in ( 'Softmax')] #'MobilenetV1/Predictions/Softmax'
    
    print("output tensor:",tensors[0])
    #exportedModel.pb
    freeze_graph.freeze_graph(FLAGS.checkpoint_path+'graph.pbtxt', "", False, 
                        FLAGS.checkpoint_path+'model.ckpt-'+FLAGS.step, tensors[0],
                        "save/restore_all", "save/Const:0",
                        FLAGS.output_file, True, ""  
                        )

if __name__ == '__main__':
  tf.app.run()
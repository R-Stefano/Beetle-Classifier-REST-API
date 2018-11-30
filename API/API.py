import os
import numpy as np
import tensorflow as tf
import flask
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2


app=flask.Flask(__name__)


app.config['UPLOAD_FOLDER'] = "static"
classesFile="classes.txt"
classesString=[]

def prepareClasses():
    global classesString
    #Load possible classes
    o=open(classesFile, "r")
    classes=o.read().split()
    o.close()

    #Save classes in a list
    for cl in classes:
        classesString.append(cl)
    
    print("classes loaded")

def loadModel():
    #def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile("exportedModel.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
        
        # We can verify that we can access the list of operations in the graph
        #for op in graph.get_operations():
            #print(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions
        
        x = graph.get_tensor_by_name('import/fifo_queue_Dequeue:0')
        y = graph.get_tensor_by_name('import/MobilenetV2/Predictions/Softmax:0')

        sess=tf.Session(graph=graph)
    print("Model loaded")
    return sess, x, y


@app.route("/upload", methods=["POST","GET"])
def upload():
    if request.method=="POST":
        file=request.files['photo']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        img=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img=cv2.resize(img[:,:,::-1], (224,224))

        result=sess.run(y, feed_dict={x:np.repeat(np.expand_dims(img, axis=0), 32 ,axis=0)})

        #The image is processed with data augmentation to obtain an ensable of results.
        avg_pred=np.mean(result, axis=0)
        
        #return the data dictionary as a JSON response
        return render_template("index.html", specie=classesString[np.argmax(avg_pred)].replace("-", " "), image=filename)
    else:
        return render_template("index.html", data="world")

	
if __name__ == "__main__":
    prepareClasses()  
    sess,x,y=loadModel()
    app.run()
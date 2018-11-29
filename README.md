# Classify Beetle species using pre-trained Resnet

## 1 - Create TFRecords
Dataset contains 10 folders, one for each beetle specie. Each beetle specie has around 200 images that we took from google images using a simple javascript script.

In order to convert the original images to TFRecords, run 

```
	python generate-TFFormat.py
```

Once finished, it will create the TFRecords in the folder ```data```

## 2 - Allows fine-tune on custom dataset
In order to fine-tune the model on our dataset, I need to develop a *data provider* and add it to *slim src code*. In order to achieve this:

### Create my data provider
models/research/slim/datasets/ stores the datasets that can be used to train the network. So, I modified ```flowers.py``` to accomodate mine which I called *beetle.py*. It is in the same folder.

### Add the data provider to dataset_factory.py
models/research/slim/datasets/dataset_factory.py has been modified in order to accomodate our dataset.
In ```dataset_factory.py```, I simply added our dataset name 'beetle' which links to the beetle.py file just created.

Now, we are ready to train the model.

## 3 - Download and prepare the pre-trained model
From (https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)[Here] you can download one of the pre-trained models.

We downloaded *ResNet V1 50*, unzipped and stored in a folder called *resnet_50*. In *models/research/slim*.

### Which variables train?
In ```train_image_classifier.py``` at line 358, after ```for var in slim.get_model_variables():```.
You can add a ```print(var)``` to display all the variables that the model loaded has.

## 4 - Run the training 
In models/research/slim/ create a folder called *myModel* which is going to store the new trained model and the tensorboard logs.

In models/research/slim/ create a folder called *data* which stores the *TFRecords* created before.
from models/research/slim/, run:
```
python train_image_classifier.py \
    --train_dir=myModel \
    --dataset_dir=data/ \
    --dataset_split_name=train \
    --dataset_name=beetle \
    --model_name=mobilenet_v1 \
    --checkpoint_path=mobilenet/mobilenet_v1_1.0_224.ckpt \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    --trainable_scopes=MobilenetV1/Logits \
    --clone_on_cpu=True \
    --save_interval_secs: 60

```

You can also specify:
* --optimizer: "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop"
* --save_interval_secs: The frequency which the model is saved, in seconds.

***Note:*** It automically applies data augmentation. In fact, passing as *model_name=mobilenet_v1*,
it uses *models/research/slim/preprocessing/inception_preprocessing.py* to augment the data.

## 5 - Run the evaluation
```
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=myModel/model.ckpt-<Global step> \
    --dataset_dir=data/ \
    --dataset_name=beetle \
    --dataset_split_name=test \
    --model_name=mobilenet_v1
    --eval_dir=myModel/eval
```

## 6 - Export the trained model
```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --output_file=exportedMobilenet_v1.pb
```

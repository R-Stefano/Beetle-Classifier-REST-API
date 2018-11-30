# REST API - Beetle Classifier 
<p align="center">
  <img width="auto" height="auto" src="https://github.com/R-Stefano/beetleDetection/blob/master/res-img.png">
</p>

# Introduction
Every year, TUI organizes an Hackathon of 24 hours in our University. This year, one of the scenarios required to develop an application to classify different species of Beetle using only images that the guests provided. 

We decided to develop a REST API using Flask that runs an Image classifier fine-tuned on a *MobileNet V2*. The model achieved 80% accuracy on test dataset on 10 different species. 

# Guideline
## 0 - Setup
Follow the guide [Here](https://github.com/tensorflow/models/tree/master/research/slim#Install)

## 1 - Create TFRecords
```dataset``` contains 10 folders, one for each beetle specie. Each beetle specie has around 200 images that we took from google images using a simple javascript script.

In order to convert the original images to TFRecords, create the folder **data** and run:

```
	python generate-TFFormat.py
```

Once finished, the TFRecords will be in the folder **data**.

## 2 - Allows fine-tune on custom dataset
In order to fine-tune the model on our dataset, I need to develop a *data provider* and add it to *slim src code*. To achieve this:

### 2.1 - Create my data provider
*models/research/slim/datasets/* stores the datasets that can be used to train the network. So, I modified ```flowers.py``` to accomodate mine which I called ```beetle.py```. It is in the same folder of ```flowers.py```.

### 2.2 - Add the data provider to dataset_factory.py
*models/research/slim/datasets/dataset_factory.py* has been modified in order to accomodate our dataset.
In ```dataset_factory.py```, I simply added our dataset name 'beetle' which links to the beetle.py file just created.

Now, we are ready to train the model.

## 3 - Download and prepare the pre-trained model
From [Here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) you can download one of the pre-trained models.

We used *MobileNet_v2_1.4_224* which seems to offer the best trade-off accuracy-speed, unzipped and stored in a folder called **mobilenet**. In *models/research/slim*.

### Which variables train?
In ```train_image_classifier.py``` at line 358, after ```for var in slim.get_model_variables():```.
You can add a ```print(var)``` to display all the variables that the loaded model has.

## 4 - Run the training 
In *models/research/slim/* create a folder called **myModel** which is going to store the new trained model and the tensorboard logs. Inside it, create also a folder called **eval** which will store the tensorboard logs for the model evaluation.

In *models/research/slim/* create a folder called **data** and move the the *TFRecords* created before in it.

Finally, from *models/research/slim/*, run:
```
python train_image_classifier.py \
    --train_dir=myModel \
    --dataset_dir=data/ \
    --dataset_split_name=train \
    --dataset_name=beetle \
    --model_name=mobilenet_v2_140 \
    --checkpoint_path=mobilenet/mobilenet_v2_1.4_224.ckpt \
    --checkpoint_exclude_scopes=MobilenetV2/Logits \
    --trainable_scopes=MobilenetV2/Logits \
    --clone_on_cpu=True \
    --save_interval_secs=300 \
    --save_summaries_secs=60

```

You can also specify:
* --optimizer: "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop"
* --save_interval_secs: The frequency which the model is saved, in seconds.
* --save_summaries_secs: The frequency which the tensorboard logs are saved, in seconds.
* --clone_on_cpu: Set to True to run on cpu if you don't have a gpu

***Note:*** It automically applies data augmentation. In fact, passing as *model_name=mobilenet_v2_140*,
it uses *models/research/slim/preprocessing/inception_preprocessing.py* to augment the data.

## 5 - Run the evaluation
If you want the evaluation while you are training the model, overwrite ```eval_image_classifier.py``` with mine. Then, always from *models/research/slim/*, run: 
```
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=myModel/ \
    --dataset_dir=data/ \
    --dataset_name=beetle \
    --dataset_split_name=test \
    --model_name=mobilenet_v2_140 \
    --eval_dir=myModel/eval/ 
```

During the evaluation the training stops.

If outOfMemory, try to reduce batch size adding the flag ```--batch_size=``` (default is 100)

## 6 - Export the trained model
Once the training is finished, move **export_graph.py** into *models/research/slim/* and run:
```
python export_graph.py \
    --checkpoint_path= myModel/ \
    --step=1640
```

It will create a file called **exportedModel.pb** which is the frozen graph.

## 7 - Run the API
Put the image that you want to predict inside **static** folder. Then, serve the Flask API calling
```
python API.py
```

Open your browser and go to ```localhost:5000/upload```.


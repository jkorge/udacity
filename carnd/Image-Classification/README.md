# Object Detection - Traffic Lights
This repo was used as a part of the capstone project for Udacity's Self-Driving Car Engineer Nanodegree. The objective was to obtain a model which could successfully identify and classify a traffic light from images captured with a car-mounted camera. Much of this work was completed following the lead of [this repo][alexLechner repo].


## Table of Contents

1. [Requirements](#Requirements)
2. [Setup](#Setup)
   - [Conda Env](#Conda-Env)
   - [Data](#Data)
   - [Models](#Models)
   - [Config Setup](#Config-Setup)
3. [Training](#Training)
4. [Appendix](#Appendix)

## Requirements
- [Anaconda 3][anaconda]
- Image Classification Model
  - This README goes through a setup using Tensorflow 1.4 and a specific revision of the Tensorflow Object Detection API. Some newer models may not be compatible with this. Check out the Tensorflow [model zoo][model zoo] for a listing of the latest image detection and classification models.
  - The following model was finetuned into the traffic light classifier and is used in this README as the primary example:
    - [ssd_inception_v2_coco_11_06_2017][ssd_inception_v2_coco_11_06_2017]
- Image data to finetune the model(s). [Here's][coldKnight dataset] the repo where labeled training data was obtained for this project

## Setup
### Conda Env

1. To set up an environment for this project run
```
conda create --name imgClassifier
conda activate imgClassifier
```
2. Set up tensorflow with pip:
```
pip install tensorflow-gpu==1.4
```
Windows users can use pip to install other dependencies
```
pip install pillow lxml matplotlib
```
Linux users will need to install the following packages with apt-get
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
```
3. Create a directory called tensorflow and clone the object detection API. Make sure to get the right revision number:
```
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
cd models
git checkout f7e99c0
```

***Windows***

4. Download v3.4.0 of protoc-zip from [here][protoc releases] and extract the executable file. While in `tensorflow/models/research` run:
```
<path/to/protoc.exe> object_detection/protos/*.proto --python_out=.
```
A for loop may be necessary to run the command:
```
for /F %F in ('dir/b objected_detection\protos\*.proto') do <path\to\protoc.exe> object_detection\protos\%F --python_out=.
```
You can run `python builders/model_builder_test.py` to validate that the preceding worked

5. In order for a python kernel to access the object detection API scripts, you'll need to add some directories to your PYTHONPATH environment variable. Instead of editing system variables you can create a short batch script in the anaconda environment's directory which will define new variables whenever the environment is activated. Start by moving into this directory and opening a new script in notepad:
```
cd %CONDA_PREFIX%
notepad etc\conda\activate.d\env_vars.bat
```
The following lines will save the current PYTHONPATH for later use and define a new value for PYTHONPATH:
```
@echo off
set OLDPYTHONPATH=%PYTHONPATH%
set PYTHONPATH=";<path\to\Anaconda3>;<path\to\tensorflow>\models;<path\to\tensorflow>\models\research;<path\to\tensorflow>\models\research\object_detection;<path\to\tensorflow>\models\research\slim"
```
A similar script should be created to return your running command line's PYTHONPATH value to its global setting:
```
notepad etc\conda\deactivate.d\env_vars.bat
# the script should contain:
@echo off
set PYTHONPATH=%OLDPYTHONPATH%
```
Be sure to deactivate and activate your conda environment to ensure these changes take effect

***Linux***

4. With protoc installed in step 2, you can now run:
```
protoc object_detection/protos/*.proto --python_out=.
```
5. Again as in the Windows instructions, environment variables will need to be setup for the object detection API.
```
cd $CONDA_PREFIX
touch etc/conda/activate.d/env_vars.sh
touch etc/conda/deactivate.d/env_vars.sh
```
Open the new scripts in vim and add the following:
```
###############
# in activate.d
###############
#!/bin/bash
export OLDPYTHONPATH=$PYTHONPATH
export PYTHONPATH=<path/to/Anaconda3>:<path/to/tensorflow>/models:<path/to/tensorflow>/models/research:<path/to/tensorflow>/models/research/object_detection:<path/to/tensorflow>/models/research/slim
###############
# in deactivate.d
###############
#!/bin/bash
export PYTHONPATH=$OLDPYTHONPATH
```
Be sure to deactivate and activate your conda environment to ensure these changes take effect
You can run `python <path/to/tensorflow>/models/research/object_detection/builders/model_builder_test.py` to validate that the preceding worked

### Data
The data described used in this example is already annotated with corresponding tfrecords. For an example of how to create annotations files and tfrecords with the laRA Dataset see the appendix.

[Download][coldKnight dataset] the dataset into a `data` directory

### Models
There are a number of pre-trained models available in Tensorflow's [model zoo][model zoo]. Many of the newer version of these models may not be compatible with the configs and scripts used in this repo. As a reference here are some links to models which _are_ compatible:
- [ssd_inception_v2_coco_17_11_2017][ssd_inception_v2_coco_17_11_2017]
- [ssd_inception_v2_coco_11_06_2017][ssd_inception_v2_coco_11_06_2017]
- [faster_rcnn_inception_v2_coco_2018_01_28][faster_rcnn_inception_v2_coco_2018_01_28]
- [faster_rcnn_resnet101_coco_11_06_2017][faster_rcnn_resnet101_coco_11_06_2017]

Download a model suited to the project needs into a `models` directory. The corresponding config file should be downloaded to a `config` directory. You can find the config files for the above models (and more) [here][model configs].

### Config Setup
Some changes to make in these configs include:
- `num_classes` should be the same as the number of classes in your dataset (4 in this case)
- `num_steps` determines how many training steps the finetuning process will take. 20,000 has been seen to produce good results. Using fewer steps, the model won't be as accurate but will train faster.
- `batch_size` will depend on the hardware available for training. A low value is recommended (3 was used in this project)
- `max_detections_per_class` and `max_total_detections` should both be reduced to 10
- `fine_tune_checkpoint` should point to the model intended for training (eg. `models/ssd_inception_v2_coco_11_06_2017/model.ckpt`)
- `input_path` for train and eval inputs should point to the training and evaluation datasets respectively. If you do not have eval data (the linked dataset does not), you can remove the eval sections from the configs
- `label_map_path` in both the training and eval sections should point to the `.pbtxt` file containing a description of the labels for your dataset (see the appendix for more on this)
- ***For faster_rcnn_inception_v2*** set the `min_dimension` and `max_dimension` to the minimum height and maximum width of your images, respectively. Using the linked dataset:
  - 600 and 800 for the simulator data
  - 1368 and 1096 for the real data

Other parameters may also be changed depending on the needs of the finetuning process. Play around (if you have time to keep retraining) and see what works best!

## Training

For convenience the necessary scripts from the Object Detection API are in the root directory of this repo. With the environment setup described, these can be run from a terminal in this root directory.

To retrain the downloaded model on the dataset, simply run:

***Windows***
```
python train.py ^
--train_dir=.\models\retrained\<model_name> ^
--pipeline_config_path=.\config\<model_name>.config
```

***Linux***
```
python train.py \
--train_dir=./models/retrained/<model_name> \
--pipeline_config_path=./config/<model_name>.config
```

This process will take a while. Once completed, you can freeze the inference graph for later use by running:

***Windows***
```
python export_inference_graph.py ^
--input_type image_tensor ^
--pipeline_config_path .\config\<model_name>.config ^
--trained_checkpoint_prefix .\models\retrained\<model_name>\model.ckpt-<num_steps> ^
--output_directory .\models\retrained\<model_name>\frozen
```

***Linux***
```
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ./config/<model_name>.config \
--trained_checkpoint_prefix ./models/retrained/<model_name>/model.ckpt-<num_steps> \
--output_directory ./models/retrained/<model_name>/frozen
```

Here `<model_name>` is the name of the model as you saved it and `<num_steps>` is the value you chose for the number of training steps in your config file. This can be replaced with earlier checkpoints that the training script creates during the finetuning process (you can even set your num_steps very high ~500,000 and freeze all the different checkpoints for later use/analysis). You'll now have a `frozen_inference_graph.pb` file you can use to run inference on new, unseen images!



--------------------------------------------------------------------------------
## Appendix
As an example of how other datasets may be used to finetune these models, a description is provided of how to create [tfrecords][tfrecords] for the  [laRA Traffic Light Dataset][laRA dataset]. This process is based on [this article on creating tfrecords][article on creating tfrecords] and this [repo][create annotations and records repo].

### Trim extraneous data from ground truth text file
The included script [to_annotations.py][to_annotations.py] can be used to convert the ground truth values from a txt file to a yaml file which can then be used to produce tfrecords. The ground truth text file ([download here][laRA ground truth]) provided with the data needs some preprocessing to remove some superfluous data and comments. The provided [lara_groundtruth_preprocess.py][lara_groundtruth_preprocess.py] will remove these fields and save a space-delimited sequence of vectors containing frame index, bounding box coordinates (x1, x2, y1, y2), and object class (ie. traffic light color) to a text file called 'ground_truth.txt'. This script should be run in the same directory as the [Lara_UrbanSeq1_GroundTruth_GT.txt][laRA ground truth] file.

### Create annotations file
Now that the ground truth values are saved in a simplified format, they can be used to create an annotations file suitable for generating tfrecords. This is a simple matter of running the provided [to_annotations.py][to_annotations.py] script:
```
python to_annotations.py --output_path <path/to/annotations.yaml> --ground_truth_path <path/to/ground_truth.txt>
```
This will read in the trimmed ground truth text file and put out the annotations yaml.

### Create tfrecords
Finally, the tfrecords can be created. These files will be used by the finetuning process to train the chosen model on the laRA images. Similar to creating the annotations yaml, tfrecords can be created with the provided [create_tfrecords.py][create_tfrecords.py] script:
```
python data_conversion --input_yaml input_file_name.yaml --output_path output_file_name.record
```
This script has been modified from [this source][create annotations and records repo] so that the `LABEL_DICT` maps to the labels used in this example. Once the tfrecords have been created, the config for the chosen model can be updated as previously described.

Note that there is no eval dataset in this repo. If it is desired to run eval on the model, make sure to include the appropriate values from the model's config. The steps in this appendix should also be repeated for the portion of the data which is to be set aside for eval. Alternatively, the provided scripts can be edited to randomly split the data into training and eval sets while creating annotations files.

--------------------------------------------------------------------------------
[//]: # (References)
[laRA dataset]: http://www.lara.prd.fr/benchmarks/trafficlightsrecognition
[laRA ground truth]:http://s150102174.onlinehome.fr/Lara/files/Lara_UrbanSeq1_GroundTruth_GT.txt
[article on creating tfrecords]: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d
[tfrecords]: https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
[ssd_inception_v2_coco_17_11_2017]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
[ssd_inception_v2_coco_11_06_2017]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
[faster_rcnn_inception_v2_coco_2018_01_28]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
[faster_rcnn_resnet101_coco_11_06_2017]: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
[model configs]: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
[model zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[coldKnight dataset]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset
[protoc releases]: https://github.com/google/protobuf/releases
[alexLechner repo]: https://github.com/alex-lechner/Traffic-Light-Classification
[anaconda]: https://www.anaconda.com/download/
[create annotations and records repo]: https://github.com/oflucas/Traffic-Light-Detection
[create_tfrecords.py]: ./create_tfrecords.py
[to_annotations.py]: ./to_annotations.py
[lara_groundtruth_preprocess.py]: ./data/lara_groundtruth_preprocess.py

# Traffic Sign Recognition
#### James Korge
-------------------------------------------------------------------------------------------
The following is an overview of the IPython Notebook (Traffic_Sign_Classifier.ipynb) included in this project submission. The notebook runs through an exploration of the images found in the German Traffic Signs Dataset provided for this project and uses both the information gathered from that analysis and methods derived from previously published work to construct a Convolutional Neural Network (CNN) capable of classifying (with 80% accuracy) a small set of previously unseen traffic sign images.

## Data Set Summary & Exploration
After loading the data files, a few high-level features were read-in using numpy’s ‘shape’ and ‘unique’:

1. Number of training examples = 34799
2. Number of validation examples = 4410
3. Number of testing examples = 12630
4. Image data shape = (32, 32, 3)
5. Number of classes = 43

This was supplemented by two visualizations of the dataset. The first was a sampling of 9 random images and their corresponding labels. This was done to observe the timbre of the images and begin ideating on how best to pursue preprocessing. The images were loaded into subplots using matplotlib and the labels were loaded into a pandas dataframe before being read into the subplot titles.

![Figure 1](images/sample-images.png?raw=true "Sample of 9 images chosen at random from the original dataset")

The second visualization was a bar chart indicating the most significant color in each pixel. This chart was made by iterating over each pixel of each image, getting the index of the max value  cross the three color-channels, and adding 1 to that channel’s count.

![Figure 2](images/color-dist.png?raw=true "Distribution of colors in the dataset. Images were scanned to count the most significant color in each pixel.")

The result of this indicates a high prevalence of red and blue colors across the data set. Upon inspection, the dataset derives its few, green-dominant pixels from background pixels where there is occasionally green foliage. It is expected that any model trained on these color images may be ill-suited to classifying green street signs which are common in certain regions (eg. the United States).

## Model Design & Testing
### Preprocessing
The first step taken in preprocessing the images was to convert them to grayscale. The images have a great deal of color bias (Figure 2) and, in some cases, are extremely dark which tends of obfuscate many features of the images. The conversion addresses these. More importantly, however, converting to grayscale reduces the number of input elements by 2 times the number of pixels which simplifies and speeds up the learning process.

![Figure 3](images/grayscale-sample.png?raw=true "Examples of images before and after converting to grayscale. Some of the images are very dark before the conversion which can make identifying features (eg. edges) more difficult")

After converting to grayscale, the images were normalized with a simple averaging: pixelValue=((pixelValue-maxPixelValue))/maxPixelValue. This further enhanced the brightness and edge clarity of the images as shown in Figure 4.

![Figure 4](images/preprocessed.png?raw=true "Sample image alongside its grayscale and normalized counterparts")

### Model Architecture
In the notebook, there are two models. The first is a LeNet-5 architecture as used elsewhere in the course. The second is based on the architecture used in Sermanet & LeCun’s follow-up research, provided in the course as extra reading (doi:10.1109/ijcnn.2011.6033589). After testing it was determined that the second (modified LeNet) architecture returned a consistently greater validation accuracy. 

LeNet:

| Layer           | Description                                             | Input Size  | Output Size |
| --------------- |:-------------------------------------------------------:|:-----------:|:-----------:|
| Convolutional   | Valid padding & 1x1 stride                              | 32x32x1     | 28x28x6     |
| Activation      | ReLU                                                    | 28x28x6     | 28x28x6     |
| Pooling         | Max pool with valid padding and 2x2 stride              | 28x28x6     | 14x14x6     |
| Convolutional   | Valid padding & 1x1 stride                              | 14x14x6     | 10x10x16    |
| Activation      | ReLU                                                    | 10x10x16    | 10x10x16    |
| Pooling         | Max pool with valid padding and 2x2 stride              | 10x10x16    | 5x5x16      |
| Flatten         |                                                         | 5x5x16      | 400         |
| Fully Connected | Weights sampled from random distribution; bias set to 0 | 400         | 120         |
| Activation      | ReLU                                                    | 120         | 120         |
| Fully Connected | Weights sampled from random distribution; bias set to 0 | 120         | 84          |
| Activation      | ReLU                                                    | 84          | 84          |
| Fully Connected | Weights sampled from random distribution; bias set to 0 | 84          | 43          |



Modified LeNet:

| Layer           | Description                                             | Input Size       | Output Size |
| --------------- |:-------------------------------------------------------:|:----------------:|:-----------:|
| Convolutional   | Valid padding & 1x1 stride                              | 32x32x1          | 28x28x6     |
| Activation      | ReLU                                                    | 28x28x6          | 28x28x6     |
| Pooling         | Max pool with valid padding and 2x2 stride              | 28x28x6          | 14x14x6     |
| Convolutional   | Valid padding & 1x1 stride                              | 14x14x6          | 10x10x16    |
| Activation      | ReLU                                                    | 10x10x16         | 10x10x16    |
| **Pooling**     | Max pool with valid padding and 1x1 stride              | 10x10x16         | 5x5x16      |
| Convolutional   | Valid padding & 1x1 stride                              | 5x5x16           | 1x1x400     |
| **Activation**  | ReLU                                                    | 1x1x400          | 1x1x400     |
| Flatten         | Flattens output of layers in **bold**                   | 5x5x16 ; 1x1x400 | 400 ; 400   |
| Concatenation   | Concatenates both flattened outputs                     | 400 ; 400        | 800         |
| Dropout         |                                                         | 800              | 800         |
| Fully Connected | Weights sampled from random distribution; bias set to 0 | 800              | 43          |


The final model concluded its last epoch of training with the following results:
* Training Accuracy = 0.999
* Validation Accuracy = 0.959
* Test Accuracy = 0.938
Initially, the LeNet-5 architecture was chosen as the model for this project; however, it was more susceptible to overfitting – training accuracy would approach 1.000 while validation accuracy was generally upper bound by 0.920. This was then substituted for the modified LeNet model.

The modified LeNet architecture was, with the following parameters, able to achieve the aforementioned results:
* Batch Size = 64
* Epochs = 20
* Learning_rate = 0.001
Although this model still exhibited overfitting, the disparity between training and validation was less than that exhibited by the LeNet-5 model.
The key differences between the LeNet-5 architecture the modified LeNet are the use of a third convolutional layer and the concatenation between the second and third activated convolutional outputs. Here it must be admitted that the theory behind why this model works better is unclear. Although Sermanet & LeCun provide clear and precise analysis of what they did and how, it is not immediately obvious (to a novice) why.
In selecting the parameters, a simple trial-and-error method was employed. Trials ran through batch sizes (64, 128, 256, 512, 1024), number of epochs (10, 20, 50), and learning rates (0.1, 0.01, 0.001, 0.0001) independently. The final values formed a superset of the most successful parameters.

## Classifying New Images

The final stage of this project consisted of using the model to classify 5 new images taken from the web. These images were subject to the same preprocessing as the data used to train/test the model.

![Figure 5](images/new-images.png?raw=true "New traffic sign images taken from the web. The second and third rows show the images after converting to grayscale and then normalizing, respectively")

The last sign (Speed Limit 70) was suspected be difficult to classify as that image contains more negative space than may of the images in the training set. The first image (Road Work) was also suspected to present some difficulty due to the apparent glare on the apex of the sign.

### Results

Upon passing these images through the model, the only image that was incorrectly classified was sign 3 (Right-fo-way at the next intersection). This constitutes an accuracy of 80% compared to the 93% on the test set. Although this difference seems large, this round of testing only contained 5 new images – 0.04% the number of images in the test set. As such, it is difficult to conclude how favorably this model will accept new data with further, more rigorous testing.

Classification Results:

| Header One                            | Header Two                            |
| ------------------------------------- | ------------------------------------- |
| Road Work                             | Road Work                             |
| No Entry                              | No Entry                              |
| Children Crossing                     | Dangerous Curve to the Right          |
| Right-of-Way At The Next Intersection | Right-of-Way At The Next Intersection |
| Speed Limit (70 km/h)                 | Speed Limit (70 km/h)                 |

There was a high degree of certainty in predicting the 4 correctly-predicted images. This is determined by the top-3 softmax probabilities.

IMAGE | 1ST SOFTMAX | 2ND SOFTMAX | 3RD SOFTMAX | 4TH SOFTMAX | 5TH SOFTMAX
--- | --- | --- | --- | --- | --- |
ROAD WORK | 0.992 | 0.00792 | 1.4e-6 | 8.33e-7 | 4.58e-7
NO ENTRY | 1.00 | 2.13e-25 | 7.83e-26 | 2.93e-27 | 1.77e-29
CHILDREN CROSSING | 0.890 | 0.0831 | 0.0241 | 1.16e-3 | 7.38e-4
RIGHT-OF-WAY AT THE NEXT INTERSECTION | 1.00 | 7.32e-17 | 1.04e-19 | 3.36e-20 | 6.05e-21
SPEED LIMIT (70 KM/H) | 1.00 | 1.42e-8 | 5.02e-14 | 1.36e-17 | 1.67e-20


On the correctly classified images, the degree of certainty was 100% (or nearly so) in all cases. However, there is a sharp disparity in the top-3 probabilities of the Children Crossing sign. In this case the top probability falls short of 90%. However, this still indicates a high degree of certainty for an incorrect classification. Upon investigation, it seems that the correct sign was the third most likely according to the model (Figure 6). In every other case, the model’s most probable image was correct.

![Figure 6](images/soft-max.png?raw=true "New images and samples of the top-3 most probable classifications, according to the CNN")





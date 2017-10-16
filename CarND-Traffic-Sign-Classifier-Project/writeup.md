#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_writeup/visualization.png "Visualization"
[image2]: ./images_writeup/preprocessing.png  "Preprocess"
[image3]: ./images_writeup/augment.png "Augmented Data"
[image4]: ./customdata2/image_001.png "Traffic Sign 1"
[image5]: ./customdata2/image_002.png "Traffic Sign 2"
[image6]: ./customdata2/image_003.png "Traffic Sign 3"
[image7]: ./customdata2/image_004.png "Traffic Sign 4"
[image8]: ./customdata2/image_005.png "Traffic Sign 5"
[image9]: ./customdata2/image_006.png "Traffic Sign 6"
[image10]: ./customdata2/image_007.png "Traffic Sign 7"
[image11]: ./images_writeup/testimages.png "Test Images with ClassId"
[image12]: ./images_writeup/testimages_jittered.png "Test Images jittered"
[image13]: ./images_writeup/newimages_result.png "result"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lokesh1210/SDCND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43
* 


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across all classes. as we can see there is an uneven distribution of the number of samples per class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because and updated the local histogram using the Open cv library...

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]

As a last step, I normalized the image data because...

also along with the preprocessing data i have normalized the RGB data to give as input to the network and let the network decide on color space conversion.

To add more data to the the data set, I used the following techniques because ... 
1. Translate
2. Scale
3. Warp
4. Brightness

Here is an example of an original image and an augmented image:

![alt text][image3]



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image       						| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Drop out				| keep prob = 0.9								|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |
| Drop out				| keep prob = 0.8								|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x128   	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				    |
| Drop out				| keep prob = 0.7								|
| Fully connected input	| flatten 4x4x128 + maxpool(8x8x64) + maxpool(16x16x32)|
| Fully connected		| 1024.        									|
| Dropout	            | keep prob = 0.5								|
| Fully connected		| 43.        									|
| Softmax				| output      									|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Cross entropy loss and adam optimizer on loss function

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of  0.9999
* training set Loss of  0.000510
* validation set accuracy of  95.19
* test set accuracy of  94.70

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* I have choosen LeNet architecture and accuracies were about 90%
* What were some problems with the initial architecture?
* with the initial architecture the normalization was not double precision and accuracies are stuck at 60% maximum. after fixing the normalization process the accuracies improved to 85% and saturated about 90% with increased epochs. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* I have added additional layers as per the sermanet article and dropout layers as per navtosha the accuracy improved on validation set to 97% however a stable number is settled around 95.8%
* Which parameters were tuned? How were they adjusted and why?
* The tuning parameters include the number of layers, epochs, learning rate.and dropout
*      number of layers were added one layer after other to improve the taining accuracy untill the accuracy reached the maximum value
*      epochs. initially tried with low epochs of up to 10. however the the learning rate was higher and not  settled well to good accuracies
*      learning rate: learning rate and epochs are tuned together
*      dropout: i used a decreaing keep prob drop out layers as the layers deepen so that the basic features in first few layers detected and regularied in the  deeper layers
*      
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* The convolution layer uses the spatial convolutions to classify compared to a simple flatten networks. the dropout layers helps the network to no overfit and  on the training data 

If a well known architecture was chosen:
* What architecture was chosen? Sermanet and LeCun article with additional layers
* Why did you believe it would be relevant to the traffic sign application? its a convnet with similar sizes and target for traffic sign classification
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? the accuraries on traing,  validation , and test set are similar regardless of the number of images which indicates the model has generalized well and not overfit
* 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![alt text][image11]
![alt text][image12]

The images are in good lighting condition to detect. how ever with the jitter added to the input images it might be difficult to detect by the network

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| No passing      		    | No passing    								| 
| Road Work     		    | Road Work 									|
| Right of Way			    | Right of Way									|
| childern crossing	      	| Right of Way					 				|
| wild animals crossing		| wild animals crossing	     					|
| Speed Limit (80Kmph)		| Speed Limit (80Kmph)      					|
| priority road			    | priority road     							|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.71%. This compares favorably to the accuracy on the test set of 95%

Here are the results of the prediction when images are jittered:

| Image			            |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| No passing      		    | No passing    								| 
| Road Work     		    | Road Work 									|
| Right of Way			    | Right of Way									|
| childern crossing	      	| Right of Way					 				|
| wild animals crossing		| wild animals crossing	     					|
| Speed Limit (80Kmph)		| Speed Limit (80Kmph)      					|
| priority road			    | priority road     							|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.71%. This compares favorably to the accuracy on the test set of 95%

Given that the network is not trained with jittered dataset  the network performed reasonable well for new test images.  however the confidence of 100% on failure image suggests that there could be over fitting and improvements can be done 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

The top probalities of the network for jittered new images are as below
![alt text][image13]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



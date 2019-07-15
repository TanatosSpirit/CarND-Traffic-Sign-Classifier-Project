# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/bar_chart.png "bar_chart"
[image2]: ./images/data_viz.png "data_viz"
[image3]: ./images/2.jpg "2"
[image4]: ./images/12.jpg "12"
[image5]: ./images/28.jpg "28"
[image6]: ./images/34.jpg "34"
[image7]: ./images/36.jpg "36"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is divided into classes in training set.

![alt text][image1]

Here are examples from each class

![image21]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalize the image data because I wanted to simplify work for optimizer.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As a first step, I decided to shuffle the examples using shuffle from the sklearn.utils library.

As a last step, I normalized the image data to increase accuracy.

Model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	     		    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Flatten		        | output = 400        							|
| Fully connected		| output = 120 									|
| RELU					|												|
| Fully connected		| output = 84 									|
| RELU					|												|
| Fully connected		| output = 43 									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is in cell 14.

To train the model I used 20 epochs, a batch size of 128 and a learning rate of 0.001.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is in cell 14, 15.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.934 
* test set accuracy of 0.939

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

The first model is adapted from LeNet architecture. Since LeNet architecture has a great performance on recognizing handwritings, I think it would also work on classifying traffic signs.

* What were some problems with the initial architecture?

The initial architecture take grayscale image i.e. one channel image and it predicted only 10 classes. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I changed LeNet input from 32x32x1 to 32x32x3 and output from 10 to 43 according to the number of classes of signs and I used the same parameter given in LeNet lab. Its training accuracy initially was around 88-90%, then I added a dropout layer. It helps to increase accuracy to 94-95%.

* Which parameters were tuned? How were they adjusted and why?

I also tuned number of epoch to 20.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because ? .

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (50km/h)   						| 
| Priority road         | Priority road 								|
| Children crossing		| Children crossing								|
| Turn left ahead	    | Turn left ahead					 		    |
| Go straight or right	| Go straight or right      				    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set because data quality is higher. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (50km/h) (probability of 0.99), and the image does contain a Speed limit (50km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h)   									| 
| .01     				| Speed limit (30km/h) 										|
| .00					| Speed limit (60km/h)											|
| .00	      			| Speed limit (80km/h)					 				|
| .00				    | Bicycles crossing      							|


For the second image, the model is relatively sure that this is a Priority road (probability of 1.00), and the image does contain a Priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road   									| 
| .00     				| Traffic signals 										|
| .00					| Right-of-way at the next intersection											|
| .00	      			| End of no passing by vehicles over 3.5 metric tons					 				|
| .00				    | Stop     							|

For the third image, the model is relatively sure that this is a Children crossing (probability of 0.98), and the image does contain a Children crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| Children crossing   									| 
| .00     				| Road narrows on the right 										|
| .00					| Right-of-way at the next intersection											|
| .00	      			| End of no passing by vehicles over 3.5 metric tons					 				|
| .00				    | Stop     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



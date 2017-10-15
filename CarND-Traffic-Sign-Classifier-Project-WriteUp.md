**Traffic Sign Recognition** 
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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

!!!LINK!!!

###Data Set Summary & Exploration

####1. Basic summary of the data set. 

Summary statistics of the traffic signs data set using NumPy:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 as RGB image and, of course, 32x32x1 as grayscale image
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of classes. I have not added additional data to get an equal distribution of classes. Normalization and grayscaling the images showed good results. 

<img src="classes_histogram.png " width="480" alt="Combined Image" />

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc




As a first step, I decided to convert the images to grayscale because the RGB information (color) of the images did not lead to a great increase of information. Most traffic signs use the same colors like red, white and black. Of course there are some other traffic signs with other colors but overall more relevant are the shapes of the pictures than the colors.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a next and last step, I normalized the image data because ...
Data Normalization
Then I normalized the gray-scale data between 0.1 and 0.9. This step is similar to the discussion in one of the TensorFlow lectures. This will prevent the model to overfit due to the large range between the values. Additional the normalization between 0.1 and 0.9 avoids possible problems by allowing data to be 0.


####2. Final model architecture
Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used a dictionary for weights and biases to which I referred within the model. Imo this looks more clear.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32      									|
| Max pooling		| 2x2 stride, same padding, outputs 5x5x32        									|
| Fully connected layer		| Input: 800, output: 512     							|
| RELU					|												|
| Dropout layer				| keep_prob = 0.5											|	
| Fully connected layer		| Input: 512, output: 128
						|
| RELU					|												|
| Dropout layer				| keep_prob = 0.5											|	
| Fully connected layer		| Input: 512, output: 43



						
									
 

I implemented dropout layers after the fully connected layer to prevent overfitting of the model.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer and as loss function I used the softmax cross entropy with logits as cost. 

I minimized the cost with the Adam optimizer.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To get a sufficient validation accuracy I played with the hyperparameters. First I adjusted the learning rate and checked the validation accuracy. I started with way too many epochs, so I reduced the amount of epochs further and further to avoid overfitting of the model on the training data. As well, I tried several test runs with different filter depth and neurons in the fully connected layer to get a compromise between run time and accuracy.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.967
* test set accuracy of 0.948

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
--> The architecture of the network was not changed. The kept the amount of layer but I have changed the size of the network, in other words the number of parameters. Well, I assume adding a third convolutional layer will increase the accuracy further but my model achieved a sufficient performance for the project. Nevertheless, I think I will try out other model architectures, e.g. add an additional convolutional layer as well, independent from the submission.

* What were some problems with the initial architecture?
--> The first runs showed a poor performance with regards to validation accuracy. But tuning the hyperparameters and the number of model parameters helped a lot.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
--> I think this question got answered before within my project write up. :-)

* Which parameters were tuned? How were they adjusted and why?
--> Learning rate was reduced (I started with a large one)
--> Number of model parameters was increased
--> Number of epochs was incrementally decreased from 100 to 25 to avoid overfitting of the model.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
--> Convolutional layers are working fine for image classification due to the extraction of different image features by scanning the image with the filter.
--> Dropout will avoid overfitting by turning off random neurons within the model. The data has to flow "in a different way" through the model in each iteration (I know, kind of an abstract description).


If a well known architecture was chosen:
* What architecture was chosen?
--> I took the same architecture as in the TensorFlow lecture of "LeNet".
* Why did you believe it would be relevant to the traffic sign application?
--> I finished the deep learning nanodegree right before I started with the CarNd and during the DPNd I learned that such a model architecture works well for these kind of classification tasks.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
--> If the validation accuracy is in the same range as the training accuracy and the testing accuracy leads as well to high values (above 0.93 or 0.94) it is a good sign that the model did not overfit and is able to generalize on new data. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. The image titles show the correct German names of the traffic signs (I am German ;-) )

"Vorfahrtsstrasse"
<img src="/new-traffic-signs/1.jpg " width="250" alt="Combined Image" />
"Achtung Fussgaenger"
<img src="/new-traffic-signs/2.jpg " width="250" alt="Combined Image" />
"Achtung Schnee"
<img src="/new-traffic-signs/3.jpg " width="250" alt="Combined Image" />
"Gefaehrliche Kurve"
<img src="/new-traffic-signs/4.jpg " width="250" alt="Combined Image" />
"Stopp"
<img src="/new-traffic-signs/5.jpg " width="250" alt="Combined Image" />

The first image might be difficult to classify because they are not scaled to the input shape of the model. Of course that needs to be done in the first place, before preprocessing them like the training data set.

To reshape them to 32x32x3, I used the cv2.resize() function.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road		| Right-of-way at the next intersection					| 
| Pedestrians		| Pedestrians 										|
| Snow			| Snow											|
| Double curve    	| Right-of-way at the next intersection					|
| Stop			| Stop											|	


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

* First image [11 30 18 12 40]


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Right-of-way 										| 
| 0.00X     			| Snow  											|
| 0.00X				| General caution 									|
| 0.00X	      		| Priority road 					 					|
| 0.00X				| Roundabout mandatory      							|



* Second image [27 11 18 28 24]


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| 0.995        			| Pedestrians										| 
| 0.0045     			| Right-of-way  										|
| 0.00035				| General caution 									|
| 0.0X	      		| Childen crossing					 				|
| 0.0X			    	| Road narrows right									|

* Third image [30 23 28 11 29]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.780        			| Snow											| 
| 0.139     			| Slippery road										|
| 0.06				| Childen crossing									|
| 0.017	      		| Right-of-way					 					|
| 0.0X			    	| Bicycles crossing									|


* Fourth image [11 41 32  6 36]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.959        			| Right-of-way										| 
| 0.036     			| End of no-passing									|
| 0.0023				| End of all speed limits								|
| 0.0X	      		| End of speed limit (80km/h)				 			|
| 0.0X			    	| Go straight or right								|

* Fifth image [14 38 17 33  4]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| Stop											|	 
| 0.0X     			| Keep right										|
| 0.0X				| No entry											|
| 0.0X	      		| Turn right ahead				 					|
| 0.0X			    	| Speed limit (70km/h)								|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



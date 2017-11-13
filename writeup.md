## Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration

I've used simple python functions to gather information for the dataset:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### Exploratory visualization of the dataset.

Here is a sample representation for each class

![traffic signs][./images/signs.png]

Plotting the number of examples per class reviews that the data is very distorted.

![Samples per class][./images/plot1.png]

### Design and Test a Model Architecture

I've decided to extend the data by rotating the images that once flipped represent a different sing. For example left turn ahead flipped is right turn ahead. 
For signs like priority path the images cant be rotated 90% left and right to achieve 2 more images. 

Todo: The data can be extended even more by blurring, gassing, grayscalling the images in classes that don't have enough samples. Unfortunately I didn't had the time for that. 

### Preprocessing

I'be preprocess the data with zero mean by using the suggested formula (pixel - 128) / 128. My first try without normalization had very low accuracy. By adding this normalization the results ware significantly better 

#### Model architecture

I used the LeNet model from the lab and added gropout to each activation. This makes sampling less bias. 
The final model had the folloing architecture

| Layer			        | Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input 32x32x3			| The input image 								| 
| Convolution 5x5 		| 1x1 stride, valid padding, Output 28x28x6 	| 
| Relu					| Relu activation 								| 
| Dropout				| Dropout of 0.5 								| 
| maxpool 2x2			| Output 14x14x6 								| 
|:---------------------:|:---------------------------------------------:|
| Convolution 5x5		| 1x1 stride, valid padding, Output 10x10x16 	| 
| Relu					| Relu activation								| 
| Dropout				| Dropout of 0.5 								| 
| maxpool				| Output 5x5x16 								| 
| flattern				| Flattern Output 400 							| 
|:---------------------:|:---------------------------------------------:|
| matmul				| Output 120 									| 
| Relu					| Relu activation								| 
| dropout				| Dropout of 0.5 								| 
|:---------------------:|:---------------------------------------------:|
| matmul				| Output 84 									| 
| Relu					| Relu activation								| 
| dropout				| Dropout of 0.5 								| 
|:---------------------:|:---------------------------------------------:|
| matmul				| output 43 									| 

A total of 5 layars 2 convolutions and 3 fully connected.

I chose to use the LeNet model since from the lessons it was obvious it process images well. And with just switching the input data from the LeNet lab with the german traffic sings data produced 0.84 accuracy. 

### Training the model

To train the model, I used a learning rate of 0.001, 50% dropouts after the activations. Each batch size was of 128 samples and I used 40 EPOCHS

LeNet is a proven architecture that works. I've tried increasing the learning rate and playing with the drop out, but that didn't have positive effect on the accuracy

My final model results were:
* validation set accuracy of 0.949
* test set accuracy of 0.936
 

### Testing the Model on New Images

I've downloaded the following 5 images from the web to test the model as shown below.  

![Test Web Images][./images/web_signs.png]

The first image might be difficult to classify because it is shot from an angle. 
The forth image is very distorted so it might not get the correct class. 
The fifth image has a black line in the bottom, so this might cause problems

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing	| Dangerous curve to the right					| 
| No entry    			| Dangerous curve to the right					| 
| Stop					| Dangerous curve to the right					| 
| No entry	      		| Dangerous curve to the right					| 
| Priority road			| Dangerous curve to the right					| 


Unfortunately the model is very bias towards Dangerous curve to the right sign or I have a mistake in the executing

The model didn't manage to guess a single image. It even wasn't in the top 5 softmax predictinos. 

There may be several reasons to this: 
1. The dataset is still not populated evenly enough, so it needs more fake data to be generated for other signs. 
2. The images I've downloaded are way to diffrent from the test data although they look close. 
3. I haven't process the images correctly. 

### Conclusion. 
Although the model covers the project requirements of 93% accuracy, it can not be used until the predictions of the web images is fixed. 

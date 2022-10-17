# Multi Scale Context Features for Image Classification

Method based on multi scale context features for image classification. This method allows the learning process to take advantage of all the information contained in the imagens, decreasing the information loss in the learning process. Our approach use the idea of aggregating image features at different scales, taking advantage of tranfer learning, being a suitable strategy to deal with images of different sizes and small datasets.

# Code files
-context_features.py
	
	This is the code of our approach. You can use it to generate all context features and concatenate them to create the multiscale context features. We also apply the train and test of the classifier in this code.

-simple_features.py
	
	In this code, we use the approach based on simple features. We also apply the train and test of the classifier in this code.

-raw_image.py
	
	This code uses the raw_image and train/test the model without transfer learning.
  
 All the codes have a dictionary of the models, enable to use the same that we apply in the paper experiments.

# Reference

For more information about the approach: link

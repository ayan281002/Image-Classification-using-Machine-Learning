# Image Classification using Machine Learning

This repository contains project files related to Machine Learning model for Image Classification 

## Feature Extraction
Histogram of Oriented Gradients (HOG) has been used to extract features from each 32*32 image.
Reference - https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f

## Pricipal Component Analysis (PCA) has been used for dimensionality reduction and reducing number of features
Reference - https://builtin.com/data-science/step-step-explanation-principal-component-analysis

## Exploratory Data Analysis
Mean image, HOG features and PCA visualisation was performed as a part of EDA

## ML Model
Support Vector Machine (SVM) has been trained on the CIFAR-10 dataset having 60,000 images divided into 10 output classes.
The training was performed on the training data (50,000 images)
The model was evaluated using the test data (10,000 images) and an accuracy of 71% was achieved
At last, the model was trained using the entire dataset of 60,000 images and was tested on unknown image samples

Deployment

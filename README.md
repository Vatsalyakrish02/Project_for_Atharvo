#  🐶 End-to-end Multli-class Dog Breed Classification

This notebook builds an end-eo-end multi-class image classifier using TensorFlow 2.0 
and TensorFlow Hub.

## 1. Problem

Identifying the breed of a dog given an image of a dog.

When I'm sitting at the cafe and I take aphotot of a dog, I want to know what breed of dog it is.

## 2. Data
The data we're using is from Kaggle's dog breed identification competition.

https://www.kaggle.com/c/dog-breed-identification/data

## 3. Evaluation
The evaluation is a file with predictions probabilities for each dog breed of each test image.

https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

## 4. Features
Some information about the data:

* We're dessaling with images (unstructured data) so it's probably best we use deep learning/transfer learning.
* There are 120 breeds of dog (this means there are 120 different classes).
* There are around 10,000+ images in the training set (these inages have labels).
* There are around 10,000+ images in the set(these images have no labels, because we'll want to predict them).

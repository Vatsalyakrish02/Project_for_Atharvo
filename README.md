# 🐕 Dog Breed Identification Project

This repository contains the code for a machine learning model designed to classify dog breeds from images. The project is based on the Kaggle Dog Breed Identification competition.

# 📖 Overview

The goal of this project is to build a model that can identify 120 different breeds of dogs from images. It uses a Convolutional Neural Network (CNN) trained on a dataset of dog images provided by the Kaggle competition. The model is built using popular deep learning frameworks such as TensorFlow and Keras.

# 🎯 Objective

The main objectives of the project include:

Building a model that can predict the breed of a dog based on an image.
Implementing image preprocessing techniques.
Improving model performance using transfer learning and fine-tuning.

📂 Project Structure

bash

├── dog_photos/                                                           # Contains photos of dog for prediction

├── log/                                                                  # Contains TensorBoard logs

├── model/                                                                # Trained models saved here

├── templates/                                                            # HTML files for the web interface

├── test/                                                                 # Contains the dataset (train data)

├── train/                                                                # Contains the dataset (train data)

├── README.md                                                             # # Project overview (this file)

├── app.py                                                                # Main file to run the web application

├── dog-vision.ipynb                                                      # Jupyter notebooks for exploratory data analysis (EDA) and model development

├── imagenet_mobilenet_v2_130_224_classification_5.tar                    # Contains the SD-based object detection model trained on Open Images V4 with ImageNet pre-trained MobileNet V2 as image feature extractor

├── labels.csv                                                            # CSV file for labels of dog photos

├── pre_processing.py                                                     # Python file for EDA 

└── tempCodeRunnerFile.py                                                 # Python Cache file

└── requirements.txt                                                      # Python dependencies

# 🚀 Getting Started

Prerequisites
Python 3.x
TensorFlow, Keras
Flask (for web deployment)
OpenCV, Pillow (for image processing)
To install the required Python packages, run:

bash

Copy code
pip install -r requirements.txt

# Dataset

Downloaded the dataset from the Kaggle Dog Breed Identification competition.
Extract the data into the Project_for_Atharvo/ .
Training the Model
To train the model, navigate to the notebooks/ directory and run the Jupyter notebook for model training:

bash

jupyter notebook notebooks/train_model.ipynb

The notebook covers:

Data loading and preprocessing
CNN architecture and transfer learning using pre-trained models like ResNet or VGG
Model training and evaluation
Running the Web Application
The project includes a Flask-based web interface where users can upload an image of a dog, and the model will predict the breed. To run the web app:

bash

python app.py
Then, open your browser and go to http://127.0.0.1:5000/ to interact with the app.

# ⚙️ Features
Preprocessing: Data augmentation, resizing images, and normalization.
Model: CNN-based model using transfer learning.
Web Interface: Upload an image of a dog and get the predicted breed.
Transfer Learning: Incorporates models like ResNet and VGG for better accuracy.

# 📝 Model Deployment and Improvements
Fine-tuning the model for better generalization.
Testing on larger image datasets.
Optimized the Flask web app for faster predictions.

# 🔗 References
Kaggle Dog Breed Identification Competition: https://www.kaggle.com/c/dog-breed-identification
TensorFlow Documentation: https://www.tensorflow.org/
Flask Documentation: https://flask.palletsprojects.com/

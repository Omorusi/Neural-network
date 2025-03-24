# Apple vs Tomato Image Classifier
This project aims to build a deep learning model that can predict whether an image is of an apple or a tomato. The challenge is that both apples and tomatoes are red and share many visual similarities, making it harder for a model to distinguish between them. However, by training a convolutional neural network (CNN), we can leverage the model's ability to learn hierarchical features from images and effectively classify them

## Objective
The goal of this project is to train a machine learning model that can take an image as input and predict whether the image contains an apple or a tomato. This is a binary classification problem, where the model learns to differentiate between two categories: apples and tomatoes.

## Techonologies 
- PyTorch: The core library for building and training the model, providing dynamic computation graphs and GPU support.
- Torchvision: Used for image transformations like resizing, normalization, and data augmentation
- PIL (Pillow): Used for image processing tasks such as opening, resizing, and manipulating images.
- Matplotlib: Visualizes images and predictions during testing.
- Kagglehub: Downloads the Apple vs Tomato dataset directly from Kaggle.( https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification )

  ## Steps during the model
  Dataset Collection: Apple vs Tomato dataset downloaded using Kagglehub.
  
  Data Preprocessing: Images opened, resized, and converted to RGB format using PIL (Pillow).

Torchvision transforms applied: Resizing images to 224x224 pixels.

Model Architecture:

Convolutional Neural Network (CNN) used.

Layers: Convolutional Layers: Extract image features (edges, textures).

Training the Model: Model trained on the training dataset and Loss function minimizes prediction error.

Prediction: After training, model predicts “apple” or “tomato” based on input image.

Testing: Test images (not seen during training) fed into the model.


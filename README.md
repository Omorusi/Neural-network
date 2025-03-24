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
 -  Dataset Collection: Apple vs Tomato dataset downloaded using Kagglehub.
   ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182332.png?raw=true)
 - Data Preprocessing: Images opened, resized, and converted to RGB format using PIL (Pillow).
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182346.png?raw=true)
  ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182421.png?raw=true)
 - Torchvision transforms applied: Resizing images to 224x224 pixels.
  ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182432.png?raw=true)
  - Model Architecture:Convolutional Neural Network (CNN) used.

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(2048)

        self.conv10 = nn.Conv2d(2048, 2048, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(2048)

        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers
        self.fc1 = nn.Linear(2048 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)  # Output layer (2 classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01)
        
        # Residual connection for deeper layers
        residual = x
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.01)
        x += residual  # Skip connection

        x = F.leaky_relu(self.bn8(self.conv8(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn9(self.conv9(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn10(self.conv10(x)), negative_slope=0.01)

        # Adaptive pooling to maintain size
        x = self.global_pool(x)

        # Flatten feature map
        x = x.view(-1, 2048 * 7 * 7)

        # Fully connected layers with dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)  # Output layer

        return x

  - Training the Model: Model trained on the training dataset and Loss function minimizes prediction error.

  
   - Testing: Test images (not seen during training) fed into the model.
      ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182548.png?raw=true)
       ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182538.png?raw=true)
       ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182525.png?raw=true)
       ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182612.png?raw=true)
      ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20182558.png?raw=true)

##  Others NN models 
-These are some of the model of neural network that i been working on :

### ANN Basic 
ANN basic( https://github.com/Omorusi/Neural-network/blob/main/Basic_ANN(Copy).ipynb)
-I experimented with the activation function in the basic ANN model from this repository. Initially, the model used ReLU, and I wanted to see how the accuracy would change if I replaced it with Tanh.

### RNN
RNN (https://github.com/Omorusi/Neural-network/blob/main/RNN_With_Chatgpt.ipynb)
-this a model using chapgpt to see how using AI could speed up creating models.

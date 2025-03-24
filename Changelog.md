# ANN Basic 
on the nootbook i use () this nootbook to get a better idea the changes that I made where:
- Add new hidden layers to see how accurancy does the model imprpove.
- As well try tanh in the activation function to see the accurancy  of it insted of using the relu activation function .

ANN basic( https://github.com/Omorusi/Neural-network/blob/main/Basic_ANN(Copy).ipynb)
- I experimented with the activation function in the basic ANN model from this repository. Initially, the model used ReLU, and I wanted to see how the accuracy would change if I replaced it with Tanh.

Here’s the accuracy with ReLU:
- this is the accurancy by using relu
- 
   ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20221323.png?raw=true)
  
-Accurancy

 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20220804.png?raw=true)
 
This is after using tagh

   ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20221345.png?raw=true)
   
  - Accurancy
  - 
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20220830.png?raw=true)
### Added new layers 
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20230809.png?raw=true)
# CNN Apple and Tomatoes
-Added more layers to the relu activation to see if the accurancy increase ,current accurancy is 65%.Deeper network: 6 convolutional layers instead of 5.
  ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20181357.png?raw=true)
- Changed the activation from relu to LeakyReLU to see how accurancy the model will be after it. The accurancy using relu was 78% 
  ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20175229.png?raw=true)
   ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20175250.png?raw=true)
- Added another layer to see if the accurancy increase
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20181722.png?raw=true)

# Chapgpt model
-These are the hpromps used to create the model:
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20224151.png?raw=true)
### First code version 
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

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
        
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        
        # Adaptive Pooling to make fully connected layer work with any input size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(1024 * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 2)  # Output layer (for binary classification)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        x = x.view(-1, 1024 * 7 * 7)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Output layer (no activation since CrossEntropyLoss is used)
        return x

# Initialize the model
model = EnhancedCNN()
print(model)

 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20224509.png?raw=true)
 ### Issues
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20224653.png?raw=true)
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20224653.png?raw=true)
 ![image alt](https://github.com/Omorusi/Neural-network/blob/main/Screenshot%202025-03-24%20224923.png?raw=true)


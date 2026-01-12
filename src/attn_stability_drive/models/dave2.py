import torch
import torch.nn as nn
import torch.nn.functional as F

class DAVE2v1(nn.Module):
    def __init__(self, input_shape=(180, 320)):
        super().__init__()
        self.input_shape = input_shape
        
        # Batch Norm at input (common in some DAVE2 variants)
        self.bn1 = nn.BatchNorm2d(3, eps=1e-3, momentum=0.1, track_running_stats=False)
        
        # Convolutional layers
        # Standard DAVE2: 5x5 stride 2, 5x5 stride 2, 5x5 stride 2, 3x3 stride 1, 3x3 stride 1
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flatten size
        self._dummy_input = torch.zeros(1, 3, *input_shape)
        with torch.no_grad():
            x = self.bn1(self._dummy_input)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            self.flatten_size = x.flatten(1).shape[1]

        # Fully Connected layers
        self.fc1 = nn.Linear(self.flatten_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.out(x))
        return x

    def get_flattened_features(self, x):
        """
        Extract features for K-means clustering.
        Returns the output of the first FC layer (100-dim).
        """
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

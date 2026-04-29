import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        # Block 1: padding=3 keeps the spatial dimensions intact for a 7x7 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling (keeps it resolution agnostic)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        # Apply Global Average Pooling
        x = self.adaptive_pool(x)
        
        # Flatten before FC layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# 2. Pre-trained ResNet50
class PretrainedResNet(nn.Module):
    def __init__(self, num_classes=4, grayscale=True):
        super(PretrainedResNet, self).__init__()
        # Load pre-trained weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # If the images are grayscale (1 channel), modify the first conv layer
        if grayscale:
            original_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Copy weights from one of the original RGB channels to maintain pre-training benefit
            self.resnet.conv1.weight.data = original_conv.weight.data[:, :1, :, :]
            
        # Replace the final fully connected layer to output 4 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 3. Pre-trained Vision Transformer (ViT)
class PretrainedViT(nn.Module):
    def __init__(self, num_classes=4, grayscale=True):
        super(PretrainedViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.grayscale = grayscale
        
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        if self.grayscale:
            x = x.repeat(1, 3, 1, 1) # Convert [B, 1, H, W] to [B, 3, H, W]
            
        # FORCE RESIZE TO 224x224 FOR ViT COMPATIBILITY
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        return self.vit(x)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 1 CUSTOM CNN
class BrainTumorCNN(nn.Module):
    """
    A custom five-block convolutional neural network designed for tumor detection in grayscale MRI scans.
    It utilizes Kaiming initialization and adaptive pooling to extract hierarchical features before final classification.
    """
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

        # Spatial Grid Pooling (4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected Layers (512 channels * 4 * 4 = 8192)
        self.fc1 = nn.Linear(8192, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies layer-specific weight initialization to optimize gradient flow.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal for Conv layers with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal distribution for Linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
        Processes the input through five convolutional blocks, flattens the resulting feature map, and outputs class logits.
        """
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
    

# 2. PRE-TRAINED RESNET50
class PretrainedResNet(nn.Module):
    """
    A ResNet50-based architecture adapted for brain tumor classification using transfer learning from ImageNet.
    It includes a modification to handle single-channel MRI inputs by replicating them across three color channels.
    """
    def __init__(self, num_classes=4, grayscale=True):
        super(PretrainedResNet, self).__init__()
        # Load pre-trained weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.grayscale = grayscale
        
        # Replace the final fully connected layer to output 4 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Prepares the grayscale input for the ResNet50 backbone and returns the final classification probabilities.
        """
        # Convert [B, 1, H, W] to [B, 3, H, W] cleanly, preserving all original 3-channel weights
        if self.grayscale:
            x = x.repeat(1, 3, 1, 1) 
            
        return self.resnet(x)

# 3. PRE-TRAINED VISION TRANSFORMER (VIT)
class PretrainedViT(nn.Module):
    """
    A Vision Transformer (ViT-B/16) model fine-tuned for MRI classification via a modified linear head.
    The architecture uses self-attention mechanisms to capture global dependencies within the brain scan images.
    """
    def __init__(self, num_classes=4, grayscale=True):
        super(PretrainedViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.grayscale = grayscale
        
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Transforms the input tensor into the three-channel format required by the ViT backbone for classification.
        """
        if self.grayscale:
            x = x.repeat(1, 3, 1, 1) # Convert [B, 1, H, W] to [B, 3, H, W]
            
        return self.vit(x)
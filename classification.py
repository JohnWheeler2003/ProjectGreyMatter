# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define relevant variables for the ML task
batch_size = 64
num_classes = 4
learning_rate = 0.001
num_epochs = 20

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([transforms.resize(512, 512), transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root='BrainTumorImage/Training', transform=all_transforms)
test_dataset = datasets.ImageFolder(root='BrainTumorImage/Testing', transform=all_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
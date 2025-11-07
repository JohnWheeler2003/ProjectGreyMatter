# Brain Tumor Explanations

# Understanding Image Data Preprocessing

## 1. What to install into venv
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas

pip install jupyter tqdm
```
torch       -> Core PyTorch library (tensors, models, training)
torchvision -> Datasets, transforms, pretrained models
torchaudio  -> Comes with torch suite (safe to include)
matplotlib  -> For image visualization
numpy       -> General math and array operations
pandas      -> For handling any data or metrics
tqdm        -> Optional progress bars for training loops

## 2. Define Image Transformations
    - transforms.Resize((128,128))
        Resizes images to find good balance between resolution and training speed
            - 512x 512 has 262,144 pixels. 128x128 has 16384 pixels ...16 times less data per image

            - Feature Learning
                - for classification (not diagnosis), CNNs don't need to see every tiny pixel detail. The patterns distinguishing classes (shape or texture differences between tumor type) remain visible even at smaller scales

    - transforms.ToTensor
        converts image to tensor (C x H x W) and normalzies pixel values from [0,255] to [0,1]

        scales pixel values.. doesn't yet normalize to [-1,1]

        - What is a tensor: multidimensional array
            scalar -> single number
            vector -> 1D tensor
            matrix -> 2D tensor
            image -> 3D tensor

        - Why use a tensor?
            - They are easily processed by GPUs
            - Automatically differentiated (for training)
            - transformed efficiently

        This function converts PIL image or NumPy array -> PyTorch Tensor
        Additionally it scales pixel values from [0,255] to [0,1] by dividing each pixel by 255
            Before:  [255, 128, 64]
            After:   [1.0, 0.5, 0.25]


    - transforms.Normalize((0,5,),(0.5,))
        rescales data so each pixel is between [-1, 1]
        helps the gradients flow better and the network train more efficiently

        Why normalize?
            - Neural networks train better when:
                - the data has zero mean (centered around 0)
                - and a consistent scale (so no single feature dominates others)
            - essentially this stabilizes gradients and improves convergence speed
            - Without normalization, some features (like bright pixels) dominate, gradients become unstable, and training may either fail to converge or overfit poorly. 

## 3. Load datasets using ImageFolder
    - Done so that each image is automatically labeled by the folder name (glioma, meningioma, etc) so that we don't have to manually label everything

## 4. Create DataLoaders
    Why create them?
        - allow you to batch and shuffle data efficiently which is useful when training on GPU

        Prevents loading of thousands of images so that your GPU/CPU memory doesn't blow up

    batch_size dictates how many images are passed through the model per training step

    Why is shuffle true for training vs false for testing?
        Shuffle=True (for training)
            - Images are shuffled each epoch to avoid the model learning any ordering pattern (epoch being one complete pass through the whole dataset)
                - for example: if all "glioma" images came before "meningioma" the model might overfit early classes
            
        Shuffle=False (for testing)
            - When testing or validating, we want results to be reproducible and consistent, not randomized
            - we just care about measuring performance, not learning patterns, so shuffling is unnecesary

## 5 and 6 Testing
    - 5 tests whether the images are being grabbed and labeled correctly
    - 6 tests the dataloaders are working correclty by outputting:
        - Batch image tensor shape which tells us how many samples it's taking, the dimensions, and the resolution of the images
        - Batch label tensor shape tells us how many batches are being taken
        - First 5 labels verify that we are using the correct categories

## NOTE ON STRUCTURE
At this point, we could have multiple separate files to make a structure like this:

ProjectGreyMatter/
│
├── brain/                         # Your virtual environment (keep separate)
│
├── data/                          # Where your dataset folders are (Train/Test)
│
├── classification.py               # Main script to run training & testing
├── model.py                        # CNN model architecture (BrainTumorCNN)
├── utils.py                        # Helper functions (e.g., visualize images)
├── train.py                        # Training loop (optional separation)
└── sample_images.png               # Generated visualization output


And we may do this down the road, but for now we will keep it in one file to make it easier to debug and see the flow

# Understanding Convolution, Activation, and Pooling

## 1. Convolutional Layers (nn.Conv2d)
Think of convolution as a small "window" (called a filter or kernel) that slides across an image
    - Each filter is a small matrix (like 3x3 or 5x5)
    - It multiplies its values by the underlying image pixel values and sums them up
    - The result becomes one pixel in the output feature map
    - Different filters learn to detect edges, textues or shapes

Why do we use it?
    - Instead of processing each pixel individually, convolution captures spatial patterns like the outline of a tumor or texture differences in tissue
    - Example: If a filter is designed to detect vertical edges, it will ouptut high values where vertical lines appear in the image like in tumor boundaries or folds in the brain structure

## 2. Activation Function (nn.ReLU)
Most common activation function is ReLu (Rectified Linear Unit)

    ReLU(x) = max(0,x)

Why do we use it?
    - It adds non-linearity so the network is able to learn complex patterns
    - It keeps positive signals and suppresses negative ones
    - It prevents gradients from vanishing like older activatinos (sigmoid/tanh) often did

It helps the network see complex shapes instead of just straight lines or simple gradients

## 3. Pooling Layer (nn.MaxPool2d)
Pooling reduces the spatial size of the image representation. Think of it like compressing the image while keeping the most important features

Most common type is Max Pooling:
    - It takes small regions (like 2x2) and keeps only the maximum value
    - This reduces the computation and helps the model generalize (less sensitive to position)

Why do we use it?
    - Pooling makes the model more robust. It doesn't seem to matter if a tumor is slightly off-center or rotated since the network will still recognize it now

## Quiz: 
**Questions**
Convolution:
What does a convolutional layer actually do when it processes an image, and why is it useful in detecting patterns like edges or shapes?

Filters (Kernels):
If we increase the number of filters in a convolutional layer (for example, from 16 to 32), what effect does that have on what the model learns and its computational cost?

Activation (ReLU):
Why do we use a non-linear activation function like ReLU after each convolutional layer? What would happen if we didn’t use it?

Pooling:
What is the main purpose of max pooling, and how does it help the model generalize better?

Order of operations:
In a typical CNN block, why do we apply convolution → ReLU → pooling in that order, instead of pooling first or skipping ReLU?

**Answers**
1. A convolutional layer takes a small matrix and multiplies its values with the underlying pixel values and then sums them up to become one pixel in the output feature map. This is used to detect edges, textures, and various shapes since it extracts features. 
2. It increases the computational cost and makes the model generalize the image features more 
3. We use a non-linear activation function in order to learn complex patterns in the data. If we did not use it, we would be unable to capture complex patterns in data 
4. Max pooling takes the maximum feature value in a small matrix and compresses the image so it is smaller and easier to process, yet keeps all the important features. 
5. We have to extract the features first like the shapes, textures, and patterns in order to be able to learn more complex patterns. We need the feature map in order to do the pooling layer. If we skipped ReLU we would be unable to learn more complex patterns

**Feedback**
Convolutional layer:
Perfect — it slides a small filter over the image to extract spatial features like edges, curves, and textures. Each filter learns a different kind of feature automatically during training.

Filters:
Exactly — more filters = the model can learn more diverse features (like vertical lines, circular regions, tumor textures, etc.), but at the cost of more computation and memory.

ReLU:
Right on. Without non-linearity, the CNN would just act like a linear filter — no matter how deep you stack layers, it would only learn simple transformations. ReLU lets it model complex shapes and tumor structures.

Pooling:
Perfect explanation. Max pooling reduces the resolution of the feature map but keeps the strongest (most relevant) signals, which helps generalization and reduces overfitting.

Order (Conv → ReLU → Pool):
Exactly. We must extract features (Conv), then apply non-linearity (ReLU) to make sense of them, and then pool to reduce spatial size. Pooling before convolution would remove detail before the network had a chance to learn from it.

# Implementation of CNN Model

## Additional Imports to add
```bash
import torch
import torch.nn as nn
import torch.nn.functional as F
```

import torch:
    - What is it?
        - This is PyTorch's core library. It gives you tensors that can run on the GPU and automatically compute gradients
    - Why we need it?
        - We need this since for every deep learning model, it's parameters, and the training data are stored in torch.Tensor objects
    - How it works:
        - Tensors are like arrays, but they can live on the GPU (.to(device)) and record the history of operations for backpropagation
        Example:

            a = torch.tensor([1.0, 2.0], requires_grad=True)
            b = a * 3
            b.sum().backward()
            print(a.grad)  # tensor([3., 3.])

            PyTorch automatically computed the gradient of the sum with respect to a

Import torch.nn as nn:
    - What is it?
        - Provides building blocks (like Conv2d, Linear, BatchNorm2d) to define neural network layers
    - Why we need it?
        - Instead of writing every layer mathematically by hand, nn lets us declare them as objects
        - Example: 

            layer = nn.Linear(10, 5)

            This creates a layer that transforms a 10-dimensional vector into a 5 dimensional one with learned weights and biases
    - How it works?
        - Each layer stores its parameters and can be used inside an nn.Module which automatically handles parameter tracking and gradients

import torch.nn.functional as F
    - What is it?
        - A functional interface that gives you operations like relu, sigmoid, softmax, and cross_entropy
    - Why do we need it?
        - It's a lightweight way to apply functions without creating a layer object
        Example:

            x = F.relu(x)

            This applies the ReLU activation in one line
    - How it works?
        - These functions are stateless (meaning they do not have learnable paramaters). This means we can call them directly in forward()

## Class Definition
```
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
```

class BraintumorCNN(nn.Module):
    - Defines the CNN as a class that inherits from nn.Module. This gives it all PyTorch's neural network features (like tracking parameters, moving to GPU, saving, etc)
    - Standard for all custom models to inherit from nn.Module

def __init__(self):
    - This is the constructor. It contains everything that defines the structure of the model (layers, dropout, etc)

super(BrainTumorCNN, self).__init__()
    - This calls the parent nn.Module's constructor so PyTorch can register the layers properly
            
## Convolutional Block 1

nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    - in_channels=3 -> input has 3 color channels (RGB Images)
    - out_channels=32 -> Learns 32 filters, meaning it outputs 32 "feature maps"
    - kernel_size=3 -> Each filter is 3x3 pixels
    - padding=1 -> Adds 1 pixel of zero padding on each side, so the spatial dimensions(height and width) stay the same after convolution

    Conceptually:
        - Each convolutional layer slides these filters across the image which computes dot products at each location, detecting local patterns like edges or curves
    Mathematically:
        output_pixel = sum(kernel_values * image_patch) + bias

        this is repeated for each pixel and each filter

    The output after Conv2d(3 -> 32) is [batch_size, 32, height, width]

nn.BatchNorm2d(32)
    - What does it do?
        - Normalizes each channel's activations to have zero mean and unit variance.
    - Why?
        This makes training faster and more stable (reduces "internal covariate shift")
    - How?
        - During training, it computes batch mean and variance, then rescales

nn.MaxPool2d(kernel_size=2, stride=2)
    - What does it do?
        - Downsamples the image by taking the maximum value in each 2x2 region
    - Why?
        - Reduces spatial dimensions (height and width), which keeps the most dominant features, and reduces computation
    - Effect
        - Halves the spatial resolution
            If the image was 128x128, it would now be 64x64 after the first pool

## Repeated Convolutional Blocks (2-5)

self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.bn2 = nn.BatchNorm2d(64)
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

Think of each block as:
    - Going **deeper** -> doubling the number of filters (out_channel)
    - Going **smaller** -> halving the spatial dimensions each time (128 -> 64 -> 32 ->16 ->8 -> 4)

    After all five blocks:

        Input: [3, 128, 128]
        ↓
        Conv + Pool layers
        ↓
        Output: [512, 4, 4]

    That's 512 channels, each 4x4 in size

## Fully Connected (Dense) Layers

nn.Linear(512 * 4 * 4, 256)
    - The flattened output from the convolutional part has 512 * 4 * 4 = 8192 features per image
    - nn.Linear creates a layer that maps 8192 inputs -> 256 outputs
    - This mxes all features and lets the model learn higher-level combinations

nn.Dropout(0.5)
    - Randomly drops 50% of neurons in each training step
    - Forces the network not to rely on any single feature -> prevents overfitting

nn.Linear(256, 4)
    - Final layer that outputs 4 values -> one for each class
    - These aren't probabilities yet; we'll apply a softmax later (via CrossEntropyLoss)

## Forward Pass

x = self.conv1(x) -> applies filters to the input image
self.bn1(x) -> normalizes the results
F.relu() -> activates neurons (keeps positives, zeroes negatives)
self.pool1() -> downsamples 

This is repeated for 5 blocks

Then:

    x = x.view(x.size(0), -1) 

    This flattens the 4D tensor [batch, 512, 4, 4] into [batch, 8192]

Finally:
    F.relu(self.fc1(x)) 
        - dense layer + ReLU activation
    self.dropout(x)
        - applies dropout
    self.fc2(x)
        - output logits (raw class scores)

## Testing Model Output Shape
model = BrainTumorCNN().to(device)
    - Creates an instance of the model class
    - .to(device) moves it to the GPU (if it's available) so computations run faster

data_iter = iter(train_loader)
    - converts DataLoader into a python iterator so you can grab one batch manually

images, labels = next(data_iter)
    - retrives the next batch (usually 32 images and labels) from your training set

images = images.to(device)
    - moves the batch of images to GPU too and must match the model's device

outputs = model(images)
    - sends the images through the forward pass of the model
    - returns a tensor of shape [batch, 4] which are the raw scores of each of the 4 tumor classes

print("Input batch shape:", images.shape)
    - Example: [32, 3, 128, 128]
        - 32 images per batch
        - 3 channels (RGB)
        - 128 x 128 pixels

print ("Output shape:", output.shape)
    - Example [32, 4]
        - 32 predictions (one per image)
        - 4 class scores (one per tumor type)

## Questions
1. If my MRI images are greyscale, do I need 3 in_channels?
    - No, in_channels should be in_channels=1 since RGB images have 3 color channels (red, green, blue) and grayscale images only have 1 (intensity values)
    - First convolutional layer was:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        
        Should be:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    If it is left as 3, PyTorch will complain that the input has only 1 channel but the layer expects 3


2. Additionally, in each subsequent convolutional block, why is the in_channels number getting higher and higher?
    - Each convolutional layer learns *filters* that can detect patterns from the previous layer's feature maps
    - So as we go deeper:
        - Earlier layers learn simple things (edges, corners)
        - Middle layers learn combinations (textures, shapes)
        - Later layers learn concepts (tumor outlines, organ structures)
    - To give the network more expressive power as it learns more complex patterns, we increase the number of feature maps (i.e. channels)
    - That's why out_channels also grows as we get deeper into the network


3. How this is working is that you are defining the convolutional blocks and setting parameters for each convolutional, batch, and pooling layer to use, and then we are utilizing them in the forward function, correct? And in that forward function we are putting the relu function over each set why and what does it do/affect?
    - In the __init__() function, we define the *structure* (the "blueprint"):
        
        self.conv1 = nn.Conv2d(...)
        self.bn1 = nn.BatchNorm2d(...)
        self.pool1 = nn.MaxPool2d(...)

    - These lines don't run yet, but they *create the layers* and store them in the model
    - In the forward() function, we *apply* those layers to real input data, in the order we want:

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

    - This makes data flow through the network

    - ReLU (Rectified Linear Unit) -> ReLU(x) = max(0,x)
        - This means if the neuron output is negative, make it 0. If it's positive, keep it
        - Why does this matter?
            - Introduces non-linearity - without it, the CNN would be a fancy linear transformation (unable to learn complex patterns)
            - Prevents the "vanishing gradient" problem that older activation functions (like sigmoid) had
            - keeps computations simple and fast

    When `F.relu(self.bn1(self.conv1(x)))` is seen, the flow is:
        - conv1 -> extract features
        - bn1 -> normalize them
        - relu -> activate only the meaningful ones


4. What does it mean raw scores for each of the 4 tumor classes in the outputs = model(images) explanation?
    - Each image produces 4 numbers (one per class)
    - These numbers are called logits (aka raw scores)
    - These logits are not probabilities yet and they can be any real value (negative or positive)
    - Example: `tensor([-1.5, 2.3, 0.8, -0.4])`
        - The largest score (2.3 here) corresponds to the model's predicted class.
        - In order to turn these into probabilities (like "85% tumor A, 10% tumor B..."), we apply softmax (happens inside the loss function (CrossEntropyLoss)), so we don't manually add softmax in the model


5. Explain what it means by 32 predictions and 4 class scores under the print("Output shape:", outputs.shape). Is this predictions for what class it goes in? Are the scores percentages for how likely the image is to fall in each class?
    - If the batch size is 32, you pass 32 MRI images into the model at once and each image produces 4 scores which are the logits discussed in the previous question


## Sample output after these steps:
Using device: cpu
BrainTumorCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=8192, out_features=256, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=256, out_features=4, bias=True)
)
Input batch shape: torch.Size([32, 3, 128, 128])
Output batch shape: torch.Size([32, 4])

# Loss Function and Optimizer
## criterion = nn.CrossEntropyLoss()
    - Purpose:
        Creates a loss funciton - specifically cross-entropy loss which is standard for multi-class classification problems

    - What it does:
        When the model makes predictions, they're just numbers and not probabilities. Cross Entropy combines two steps internally:
            - Applies Softmax to turn logits into probabilities that sum to 1
            - Computes the negative log-likelihood of the correct class

                Loss=−N1​i∑​log(softmax(xi​)[yi​])

                xi are the model's raw outputs (logits)
                yi are the true labels

    - What it affects
        This value guides the optimizer on how far off the predictions are
            - the lower the loss -> the closer the model's predicted probabilities are to the correct labels

## optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    - Purpose:
        The optimizer is the algorithm that adjust the model's weights based on the loss.
        The goal is to minimize the loss by updating the parameters (weights and biases) during training

    - Why Adam?
        - This stands for Adaptive Movement Estimation
        - It automatically adapts the learning rate for each parameter using running averages of both gradients and squared gradients
        - Put simply, it learns how fast to learn for each euron individually, making it more stable and efficient than vanilla SGD
    
    - model.parameters()
        - passes all the trainable parameters of the CNN (every weight and bias) into the optimizer
        - Updated by the optimizer every iteration (batch) during training

    - lr=0.001
        - Learning rate controls how big of a step the optimizer takes when updating the weights
            - too high -> unstable training (overshoots minima)
            - too low -> very slow learning

            - 0.001 is standard and safe starting value for Adam

# Training and Validation Loop
## Beginning Notes
This is how the CNN actually learns

num_epochs: one complete pass through the entire training dataset
    - More epochs -> better learning *to a point*, but risk of overfitting if trained too long

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

These are to keep track of the metrics across epochs for visualization. They are plotted at the end to see how the learning evolves (loss should go down, while accuracy goes up)

The Epoch Loop:
```
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)
```
Loops through training once per epoch. Progress headers are printed for clarity in output

## Training Phase
`model.train()`
    Purpose: Puts the model in training mode
        - Enables dropout layers (randomly disables neurons to prevent overfitting)
        - Keeps Batch Normalization layers updating their running averages
    Without this
        - The model might behave differently and training accuracy wouldn't be reliable

Batch Loop
    `for images, labels in train_loader:`
        Each iteration gives you one batch of data
            - images is a tensor of shape [batch_size, 3, 128, 128]
            - labels is a tensor of shape [batch_size] with class indices(0-3)

Move tensors to device
    `images, labels = images.to(device), labels.to(device)`
        - Makes sure both tensors are on the same device (GPU if abailable, CPU otherwise)
        - Training would break if the model and data were on different devices

Zero Gradients
    `optimizer.zero_grad()`
        - Clears previous gradients. Gradients accumulate in PyTorch, so if we didn't clear them, they'd mix up with previous batches

Forward Pass
    `outputs = model(images)`
        - Feeds the batch of images through the CNN
        - results in outputs, which is a tensor of shape [batch_size, 4] 
            - 4 scores (logits) per image, one for each class

Compute Loss
    `loss = criterion(outputs, labels)`
        - Compares predictions (outputs) with true labels using CrossEntropyLoss
        - Returns a single scalar loss value (averaged across the batch)

Backpropagation
    `loss.backward()`
        - computes gradients for all model parameters (weights and biases) via *automatic differentiation*
        - Each parameter knows how much it contributed to the loss

Update Parameters
    `optimizer.step()`
        - Uses the gradients previously obtained to adjust weights and biases slightly in the direction that reduces loss
        - This is where the *learning* actually happens

Track Metrics
    ```
    running_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    ```

        - loss.item() -> converts the loss tensor to a scalar value
        - Multiply by batch size to get the total loss for all images in the batch
        - torch.max(outputs, 1) -> finds the index (class) with the highest predicted score per image
        - Compared with true labels to count correct predictions

Epoch Summary Training
    ```
    epoch_train_loss = running_loss / len(train_data)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    print(f"Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc*100:.2f}%")
    ```

        - Computes average loss/accuracy across the entire training set
        - Saves the values for plotting
        - Prints progress for each epoch

## Validation Phase
`model.eval()`
    - Switches the model into Evaluation moe
        - Disables dropouts (uses all neurons)
        - Uses fixed running averages in batch norm (no updates)
    - Ensures validation accuracy truly reflects model performance

Disable Gradients
    `with torch.no_grad():`
        - Turns off the gradient tracking which saves memory and computation since we're *not* training here, just testing model performance

Validation Loop
    - Similar to training, except we:
        - Don't call optimizer.zero_grad()
        - Don't call loss.backward() or optimizer.step()

    - We only 
        - Pass images through the model
        - Compute the loss
        - Measure accuracy

Epoch Summary (Validation)
    ```
    epoch_val_loss = val_loss / len(val_data)
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc*100:.2f}%")
    ```

        - Computes total loss and accuracy on the validation set
        - Logs the values for plotting

## Training Complete + Visualization
Why do we do this step?
    - Seeing loss and accuracy curve helps detect overfitting (training loss keeps going down but validation loss goes up) and confirm that the model is *actually learning*

Plots are saved as "training_validation_curves.png" for later viewing

## Quiz
**Questions**
1. Forward Pass & Backprop
What happens during the forward pass and backward pass in a training loop?
Why do we need both, and what would happen if we only did one?

2. Gradient Reset
Why do we call `optimizer.zero_grad()` before every batch?
What problem does it prevent?

3. Model Modes
Explain the difference between `model.train()` and `model.eval()`.
What specific components of your CNN behave differently in these two modes?

4. Loss Calculation
When we calculate the training loss like this:
`epoch_train_loss = running_loss / len(train_data)`
why do we divide by len(train_data) instead of the number of batches?

5. Validation Phase
Why do we wrap the validation code with:
`with torch.no_grad():`
and what would happen if we didn’t?

6. Accuracy Computation
What does this line do, step by step?
`_, predicted = torch.max(outputs, 1)`
Why do we take torch.max along dimension 1?

7. Overfitting Detection
If your training accuracy keeps going up, but your validation accuracy starts dropping,
what does that mean and what could you do to fix it?

8. Optimizer
Explain in plain terms what the optimizer is doing when we call:
`optimizer.step()`
How does it know how much to change the weights?

9. Epochs
If you trained for too few epochs, what might you see in your training and validation curves?
If you trained for too many, what might you see?

10. Visualization
What is the main purpose of plotting loss and accuracy curves after training?
What insights can you gain that raw numbers (like final accuracy) can’t show?

**Answers**
1. The forward pass feeds the images through the training loop in batches to learn weights. The backpropagation helps to calculate the gradients based on the weights we just learned. This allows each parameter to see how much it contributed to the total loss and which ones to update in the next step. If we only did one, then the weights would only be learned but the gradient would never be calculated so we wouldn't know which way to step to optimize the model.

2. optimizer.zero_grad() resets the gradients. They are saved between each batch, and we don't want to use the gradients from the previous batch as the starting point for this next batch since we want each batch independently to make their own gradients

3. model.train() is used in order to train the model and learn weights. model.eval() is used to evaluate the metrics and make sure the model is accurate. The difference is in the dropout and running averages in the batch norm. In model.train(), the dropout is enabled which allows it to randomly disable 50% of the neurons so that it adds an element of randomness and prevents the model from learning bad data. The running averages are allowed to change in the train as well in order to update the weights for the batch norms

4. We calculate the running loss by dividing by the length of the training data because we want it over the total number of training data samples. If we did it by the number of batches, we wouldn't see the correct ratio

5. validation is wrapped with torch.no_grad() in order to save the memory. Since we are not training in the validation set, we don't need to calculate gradients. 

6. This line selects the prediction that it believes the current image falls into based on the highest number

7. This means that the model is overfitting to the training data. In order to fix this I think we would have to implement some sort of regularization techniques, but I'm not entirely sure what

8. The optimizer is taking the next step towards the optimum set of parameters. It knows how much to change the weights based on the gradients it has learned from the backpropagation. It is reducing the loss function

9. If you trained for too many epochs, you would see overfitting to the training data. The training curve would increase in accuracy, but the validation curve would drop. If you trained for too few, the model would fail to learn well and you would see low accuracy across both the training and validation curves. 

10. Plotting loss and accuracy curves after training allows us to see if the model is overfitting the data and/or actually learning anything from it. 

**Feedback**
1. Forward & Backprop
- The forward pass computes predictions,
- the backward pass computes how much each weight contributed to the total loss (gradients).
- Without backprop, the model would have no feedback on what to change.

Extra note: The direction of weight updates comes from the negative gradient (we move against it), and the magnitude depends on the learning rate.

2. Gradient Reset
Gradients accumulate by default in PyTorch — so if we didn’t reset them, we’d effectively be adding together gradients from multiple batches, corrupting the update signal.

3. Model Modes
- model.train() enables dropout and lets batch norm layers update their running averages.
- model.eval() disables dropout and uses the stored averages instead of recalculating them.
- This ensures that when we validate or test, the model behaves deterministically.

4. Loss Calculation
Good reasoning, but let’s refine slightly.

You wrote:

“We calculate the running loss by dividing by the length of the training data because we want it over the total number of training data samples.”

We want the average loss per sample, not per batch.
However — because we add up losses from each batch during training, dividing by len(train_data) ensures the result is comparable across different batch sizes.

If we divided by the number of batches (len(train_loader)), we’d get the average loss per batch, which can differ depending on how many samples are in the last batch or if the batch size changes.
So your logic was good — it’s just that this makes the metric more data-size consistent.

5. torch.no_grad()
It turns off gradient tracking, so PyTorch doesn’t store computation graphs for each operation.
- Saves memory
- Speeds up validation
- Ensures no accidental weight updates happen.

6. torch.max(outputs, 1)
You were close, but let’s get super precise here:

outputs has shape [batch_size, num_classes] — for example, [32, 4].
Each row contains the model’s predicted logits (unnormalized scores) for each class.

torch.max(outputs, 1) means:

Take the max value and index across dimension 1 (the class dimension).

We ignore the value (_) and keep the index (which represents the predicted class).

So: `_, predicted = torch.max(outputs, 1)`

→ gives the most likely class for each image in the batch.

Your summary (“selects the prediction it believes the current image falls into based on the highest number”) is basically correct, just missing the detail about dimension and meaning of the index.

7. Overfitting
Fixes include:
    - More dropout
    - Data augmentation (making the model see more diverse examples)
    - Weight decay / L2 regularization
    - Early stopping
    - Collecting more data

8. Optimizer
The optimizer adjusts weights in the opposite direction of the gradient, scaled by the learning rate.
Each optimizer (Adam, SGD, etc.) does this differently — Adam, for instance, adapts the step size for each weight individually.

9. Epochs
Excellent — clear distinction between underfitting (too few epochs) and overfitting (too many).
Also, note: if validation accuracy plateaus or drops while training accuracy climbs, that’s a strong stop signal.

10. Visualization
The curves show trends, not just outcomes.
- You can see when overfitting starts.
- You can identify learning stagnation.
- You can adjust learning rates, epochs, or architecture intelligently.

# Testing and Model Evaluation

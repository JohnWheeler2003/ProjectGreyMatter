# Brain Tumor Explanations

# Understanding Image Data Preprocessing

## 1. What to install into the virtual environment
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas

pip install jupyter tqdm
```
- `torch`: Core PyTorch library (tensors, models, training)
- `torchvision`: Datasets, transforms, pretrained models
- `torchaudio`: Comes with torch suite (safe to include)
- `matplotlib`: For image visualization
- `numpy`: General math and array operations
- `pandas`: For handling any data or metrics
- `tqdm` : Optional progress bars for training loops

## 2. Define Image Transformations
1. `transforms.Resize((128,128))`
    Resizes images to find good balance between resolution and training speed
    - 512x 512 has 262,144 pixels. 128x128 has 16384 pixels ...16 times less data per image

    Feature Learning
    - For classification (not diagnosis), CNNs don't need to see every tiny pixel detail. The patterns distinguishing classes (shape or texture differences between tumor type) remain visible even at smaller scales

2. `transforms.ToTensor`
    - Converts image to tensor (C x H x W) and normalzies pixel values from [0,255] to [0,1]

    - Scales pixel values since it doesn't yet normalize to [-1,1]

    - What is a tensor: multidimensional array
        - scalar:  single number
        - vector:  1D tensor
        - matrix:  2D tensor
        - image:  3D tensor

    - Why use a tensor?
        - They are easily processed by GPUs
        - Automatically differentiated (for training)
        - transformed efficiently

    This function converts PIL image or NumPy array → PyTorch Tensor
    Additionally it scales pixel values from [0,255] to [0,1] by dividing each pixel by 255
        Before:  [255, 128, 64]
        After:   [1.0, 0.5, 0.25]


3. `transforms.Normalize((0,5,),(0.5,))`
    Rescales data so each pixel is between [-1, 1]
    Helps the gradients flow better and the network train more efficiently

    Why normalize?
    - Neural networks train better when:
        - the data has zero mean (centered around 0)
        - and a consistent scale (so no single feature dominates others)
    - This stabilizes gradients and improves convergence speed
    - Without normalization, some features (like bright pixels) dominate, gradients become unstable, and training may either fail to converge or overfit poorly. 

## 3. Load datasets using ImageFolder
- Done so that each image is automatically labeled by the folder name (glioma, meningioma, etc) so that we don't have to manually label everything

## 4. Create DataLoaders
1. Why create them?
    - allow you to batch and shuffle data efficiently which is useful when training on GPU

   - Prevents loading of thousands of images so that your GPU/CPU memory doesn't blow up

    ```batch_size``` dictates how many images are passed through the model per training step

2. Why is shuffle true for training vs false for testing?
    ```Shuffle=True``` (for training)
    - Images are shuffled each epoch to avoid the model learning any ordering pattern (epoch being one complete pass through the whole dataset)
        - for example: if all "glioma" images came before "meningioma" the model might overfit early classes
        
    ```Shuffle=False``` (for testing)
    - When testing or validating, we want results to be reproducible and consistent, not randomized
    - we just care about measuring performance, not learning patterns, so shuffling is unnecesary



## Note on Structure
At this point, we *could* have multiple separate files to make a structure like this:

## Project Structure

```text
ProjectGreyMatter/
│
├── brain/                  # Python virtual environment (kept separate)
│
├── BrainTumorImages/       # Dataset directory
│   ├── Testing/              # Testing MRI images (by class)
    ├── Training/             # Training MRI images (by class)
│   └── Validation/           # Validation MRI images (by class)
│
├── classification.py       # Main entry point for training & evaluation
├── model.py                # CNN architecture (BrainTumorCNN)
├── utils.py                # Helper utilities (visualization, metrics, etc.)
├── train.py                # Training loop (optional separation)
└── sample_images/          # Generated visualization outputs
```


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


# Implementation of CNN Model

## Additional Imports to add
```bash
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```import torch```:
- What is it?
    - This is PyTorch's core library. It gives you tensors that can run on the GPU and automatically compute gradients
- Why we need it?
    - We need this since for every deep learning model, it's parameters, and the training data are stored in torch.Tensor objects
- How it works:
    - Tensors are like arrays, but they can live on the GPU (```.to(device)```) and record the history of operations for backpropagation
    Example:
```
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = a * 3
        b.sum().backward()
        print(a.grad)  # tensor([3., 3.])
```
PyTorch automatically computed the gradient of the sum

```import torch.nn as nn```:
- What is it?
    - Provides building blocks (like Conv2d, Linear, BatchNorm2d) to define neural network layers
- Why we need it?
    - Instead of writing every layer mathematically by hand, nn lets us declare them as objects
    - Example: 

        ```layer = nn.Linear(10, 5)```

        - This creates a layer that transforms a 10-dimensional vector into a 5 dimensional one with learned weights and biases
- How it works?
    - Each layer stores its parameters and can be used inside an nn.Module which automatically handles parameter tracking and gradients

```import torch.nn.functional as F```
- What is it?
    - A functional interface that gives you operations like relu, sigmoid, softmax, and cross_entropy
- Why do we need it?
    - It's a lightweight way to apply functions without creating a layer object
    Example:

        ```x = F.relu(x)```

        - This applies the ReLU activation in one line
- How it works?
    - These functions are stateless (meaning they do not have learnable paramaters). This means we can call them directly in forward()

## Class Definition
```
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
```

```class BraintumorCNN(nn.Module)```:
- Defines the CNN as a class that inherits from `nn.Module`. This gives it all PyTorch's neural network features (like tracking parameters, moving to GPU, saving, etc)
- Standard for all custom models to inherit from `nn.Module`

```def __init__(self)```:
- This is the constructor. It contains everything that defines the structure of the model (layers, dropout, etc)

```super(BrainTumorCNN, self).__init__()```
- This calls the parent nn.Module's constructor so PyTorch can register the layers properly
            
## Convolutional Block 1

`nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)`
- `in_channels=3` → input has 3 color channels (RGB Images)
- `out_channels=32` → Learns 32 filters, meaning it outputs 32 "feature maps"
- `kernel_size=3` → Each filter is 3x3 pixels
- `padding=1` → Adds 1 pixel of zero padding on each side, so the spatial dimensions(height and width) stay the same after convolution

Conceptually:
- Each convolutional layer slides these filters across the image which computes dot products at each location, detecting local patterns like edges or curves

Mathematically:
    ```output_pixel = sum(kernel_values * image_patch) + bias```

- This is repeated for each pixel and each filter

The output after Conv2d(3 → 32) is [batch_size, 32, height, width]

`nn.BatchNorm2d(32)`
- What does it do?
    - Normalizes each channel's activations to have zero mean and unit variance.
- Why?
    This makes training faster and more stable (reduces "internal covariate shift")
- How?
    - During training, it computes batch mean and variance, then rescales

`nn.MaxPool2d(kernel_size=2, stride=2)`
- What does it do?
    - Downsamples the image by taking the maximum value in each 2x2 region
- Why?
    - Reduces spatial dimensions (height and width), which keeps the most dominant features, and reduces computation
- Effect
    - Halves the spatial resolution
        If the image was 128x128, it would now be 64x64 after the first pool

## Repeated Convolutional Blocks (2-5)
```
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.bn2 = nn.BatchNorm2d(64)
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
```
Think of each block as:
- Going **deeper** → doubling the number of filters (out_channel)
- Going **smaller** → halving the spatial dimensions each time (128 → 64 → 32 →16 →8 → 4)

    After all five blocks:

        Input: [3, 128, 128]
        ↓
        Conv + Pool layers
        ↓
        Output: [512, 4, 4]

    That's 512 channels, each 4x4 in size

## Fully Connected (Dense) Layers

`nn.Linear(512 * 4 * 4, 256)`
- The flattened output from the convolutional part has 512 * 4 * 4 = 8192 features per image
- ```nn.Linear``` creates a layer that maps 8192 inputs → 256 outputs
- This mxes all features and lets the model learn higher-level combinations

`nn.Dropout(0.5)`
- Randomly drops 50% of neurons in each training step
- Forces the network not to rely on any single feature → prevents overfitting

`nn.Linear(256, 4)`
- Final layer that outputs 4 values → one for each class
- These aren't probabilities yet; we'll apply a softmax later (via `CrossEntropyLoss`)

## Forward Pass

`x = self.conv1(x)` → applies filters to the input image
`self.bn1(x)` → normalizes the results
`F.relu()` → activates neurons (keeps positives, zeroes negatives)
`self.pool1()` → downsamples 

This is repeated for 5 blocks

Then:

`x = x.view(x.size(0), -1)`

This flattens the 4D tensor [batch, 512, 4, 4] into [batch, 8192]

Finally:
`F.relu(self.fc1(x))` 
- dense layer + ReLU activation

`self.dropout(x)`
- applies dropout

`self.fc2(x)`
- output logits (raw class scores)

# Loss Function and Optimizer
## `criterion = nn.CrossEntropyLoss()`
- Purpose:
    Creates a loss funciton - specifically cross-entropy loss which is standard for multi-class classification problems

- What it does:
    When the model makes predictions, they're just numbers and not probabilities. Cross Entropy combines two steps internally:
    - Applies Softmax to turn logits into probabilities that sum to 1
    - Computes the negative log-likelihood of the correct class
        ```
        Loss=−N1​i∑​log(softmax(xi​)[yi​])

        xi are the model's raw outputs (logits)
        yi are the true labels
        ```
- What it affects
    This value guides the optimizer on how far off the predictions are
    - The lower the loss → the closer the model's predicted probabilities are to the correct labels

## `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`
- Purpose:
    - The optimizer is the algorithm that adjust the model's weights based on the loss.
The goal is to minimize the loss by updating the parameters (weights and biases) during training

- Why Adam?
    - This stands for Adaptive Movement Estimation
    - It automatically adapts the learning rate for each parameter using running averages of both gradients and squared gradients
    - Put simply, it learns how fast to learn for each euron individually, making it more stable and efficient than vanilla SGD

- `model.parameters()`
    - passes all the trainable parameters of the CNN (every weight and bias) into the optimizer
    - Updated by the optimizer every iteration (batch) during training

- `lr=0.001`
    - Learning rate controls how big of a step the optimizer takes when updating the weights
        - too high → unstable training (overshoots minima)
        - too low → very slow learning

        - 0.001 is standard and safe starting value for Adam

# Training and Validation Loop
## Beginning Notes
This is how the CNN actually learns

`num_epochs`: one complete pass through the entire training dataset
- More epochs → better learning *to a point*, but risk of overfitting if trained too long

```
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
```

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

Without this:
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

- Compares predictions (outputs) with true labels using    `CrossEntropyLoss`
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

- `loss.item()` → converts the loss tensor to a scalar value
- Multiply by batch size to get the total loss for all images in the batch
- `torch.max(outputs, 1)` → finds the index (class) with the highest predicted score per image
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
    - Don't call `optimizer.zero_grad()` 
    - Don't call `loss.backward()` or `optimizer.step()`

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

# Testing and Model Evaluation
## Additional Imports to add:
```
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
```
## Configuration and checkpoint loading
```
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
    if state is not None:
        model.load_state_dict(state)
```

- ` torch.load(..., map_location=device)` loads file into CPU/GPU depending on device
- `model.load_state_dict(state)` loads weights into the model which replaces the model parameters with saved ones

```
model.to(device)
model.eval()
```
- `.to(device)` ensures model is on CPU or GPU consistent with input tensors
- `.eval()` toggles evaluatio mode to disable dropout and cause BatchNorm to use learned running mean/variance instead of batch statistics

## Running inference and collecting outputs
```
all_preds = []
all_probs = []
all_labels = []

softmax = torch.nn.Softmax(dim=1)
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)          # logits
        probs = softmax(outputs)         # convert logits → probabilities
        preds = torch.argmax(probs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
```
- outputs = model(images) returns logits (raw scores). Softmax **should not** be applied during training process when using CrossEntropyLoss, but for interpreting probabilities at test time.
- `torch.argmax(probs, dim=1)` picks class with highest probability for each sample
- we `.cpu()` and convert to numpy lists so they're easy to analyze with scikit learn and numpy

## Confusion Matrix and Classification Report
```
class_names = test_data.classes
cm = confusion_matrix(all_labels, all_preds)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
```
    
- `test_data.classes` is the ImageFolder class order which we use to map indices to class names
- `confusion_matrix` returns square matrix [i,j] = count of true class i predicted as j
- `classification_report` gives precision, recall, and F1 per class:
    - Precision = TP / (TP + FP) - of predicted positives, how many were correct?
    - Recall = TP / (TP + FN) = of true positives, how many did we detect
    - F1 = harmonic mean of precision and recall

## Per-class accuracy
```
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for idx, cls in enumerate(class_names):
    print(f"Class '{cls}': {per_class_acc[idx]*100:.2f}% ({int(cm[idx,idx])}/{int(cm.sum(axis=1)[idx])})")
```

- `cm.diagonal()` are true positives for each class
- dividing by `c.sum(axis=1)` (sum over row) gives recall per clas (i.e., fraction of that class correctly identified)

## Unnormalize and visualize misclassified examples
Since we want the original-looking images (not normalized tensors), the code tries to find the Normalize used in the test transforms and invert it. This handles both single-channel and three-channel Normalize

`def get_normalize_params(transform)`:
- searches inside compose for a Normalize object and returns its mean and std

`def broadcast_mean_std(mean, std)`:
- If normalize used a single-channel tuple, we replicate it across three channels so unnormalization works for RGB images

`def unnormalize_tensor(img_tensor, mean, std)`:
- for each channel multiply by std and add mean to invert the Normalize operation, then convert to HWC for plotting

`mis_idx = np.where(all_preds != all_labels)[0]`:
- Indices of missclassified samples. Because test_loader was created with `shuffle=False`, indexing test_data[idx] returns the *same* sample that produced all_preds[idx]. If the test loader had been shuffled, you must not index test_data like this since the mapping would not match

We iterate through the first `num_classified_to_show` misclassifications, unnormalize each image, and plot them with True/Pred labels. Seeing this often reveals whether the model fails on noisy images, label mistakes, or visually ambiguous cases

## Adding Model Checkpoint
This is a saved copy of the model's parameters (weights and biases) during training. It's usually saved when the model performs best on the validation set

This prevents overfitting from ruining a good model later in training and ensures you always keep the best performer

```
checkpoint = {
    "epoch": epoch + 1,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_accuracy": best_val_acc,
}
```

This is a dictionary containing:
- The current epoch
- The model weights (state_dict)
- The optimizer state (so we can continue training later if we want)
- The best validation accuracy so far


# Random notes:
Ensure all dependencies are installed:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy pandas
```

For reproducability:
```
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```
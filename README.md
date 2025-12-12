# ProjectGreyMatter – MRI Brain Tumor Classification (PyTorch)
## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify MRI brain images into one of four tumor categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

The model is trained using a custom dataset of MRI images organized into Training, Validation, and Testing splits.

It outputs training curves, a confusion matrix, classification metrics, and visualizations of misclassified examples.

The project includes a full Makefile-based workflow that builds a virtual environment, installs dependencies, runs the model, and cleans generated files.

## Project Structure
```
ProjectGreyMatter/
├── BrainTumorImages/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
├── README.md
├── classification.py
├── explanations.md
├── makefile
├── valcreation.py
└── verifyimgcount.py
```

## Expected Dataset Structure

Each split should contain subfolders representing the four classes:
```
Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
```
## Model Architecture

The CNN architecture consists of five convolutional blocks, each containing:
- Convolution layer
- Batch Normalization
- ReLU activation
- MaxPooling

Followed by two fully connected layers:

| Layer Type   | Details                              |
| ------------ | ------------------------------------ |
| Conv Block 1 | 3 → 32 filters                       |
| Conv Block 2 | 32 → 64 filters                      |
| Conv Block 3 | 64 → 128 filters                     |
| Conv Block 4 | 128 → 256 filters                    |
| Conv Block 5 | 256 → 512 filters                    |
| FC1          | Linear(512×4×4 → 256) + Dropout(0.5) |
| FC2          | Linear(256 → 4 classes)              |


The model uses CrossEntropyLoss and the Adam optimizer (`lr = 0.001`).

## Training Pipeline

The training loop includes:
- GPU/CPU detection
- Loss & accuracy tracking
- Validation at each epoch
- Automatic saving of the best-performing model checkpoint
- Training/validation curve visualization
- Early qualitative debugging through misclassification plots

## Training Output Files
| File                             | Description                           |
| -------------------------------- | ------------------------------------- |
| `best_brain_tumor_cnn.pth`       | Best model checkpoint                 |
| `training_validation_curves.png` | Loss & accuracy plots                 |
| `confusion_matrix.png`           | Saved confusion matrix                |
| `misclassified_examples.png`     | Sample images the model misclassified |

## Evaluation Metrics

After training, the code automatically computes:
- Test accuracy
- Confusion matrix
- Classification report
    - Precision
    - Recall
    - F1-score
- Per-class accuracy
- Misclassified sample visualization

## Reproducibility

The following techniques help ensure consistent results:
- Fixed transformation pipeline
- No shuffling in the test loader
- Checkpoint-based testing
- Deterministic evaluation mode (model.eval() disables dropout)

If exact bit-level reproducibility is required, PyTorch seeds and deterministic backend settings can be added.

## How to Run the Project

This project is fully automated using a Makefile.
1. Create the virtual environment & install dependencies

    `make`

    This creates a virtual environment named brain and installs:
    - torch
    - torchvision
    - torchaudio
    - matplotlib
    - seaborn
    - scikit-learn
    - numpy
    - pandas
    - timm

2. Run the classification pipeline

    `make run`


    This runs classification.py inside the virtual environment and automatically checks for missing packages.

3. Clean temporary files

    `make clean`
    Remove caches, .png plots, .pth checkpoints:

4. Remove the virtual environment
   
    `make clean-env`

6. Full rebuild
   
    `make rebuild`

## Key Features

- Simple but deep CNN architecture
-  GPU support (if available)
-  Automatic checkpointing
- Training & validation visualizations
-  Confusion matrix & classification report
-  Misclassified image visualization
-  Full Makefile workflow
- Reproducible, modular, easy to extend

## Future Improvements

Potential extensions include:
- Data augmentation (rotation, flip, noise)
- Learning rate scheduling
- Transfer Learning (ResNet, EfficientNet)
- Grad-CAM visualization to interpret model decisions
- Hyperparameter tuning (batch size, CNN depth)

## Authors

John Wheeler, Heriberto Rosa, Linh Luong

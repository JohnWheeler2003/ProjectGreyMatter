# ProjectGreyMatter II – MRI Brain Tumor Classification (PyTorch) (README OUT OF DATE AS OF 4/30/26)
## Overview

This project implements a Deep Convolutional Neural Network (CNN) using PyTorch to classify MRI brain images into one of four tumor categories: Glioma, Meningioma, Pituitary, or No Tumor. Currently, the project is expanding upon its base custom CNN architecture to integrate advanced deep learning techniques, including ResNet architectures, data augmentation, learning rate scheduling, hyperparameter tuning, and model interpretability to better visualize how the model makes decisions.

### Key Features:

- Native hardware acceleration optimized for Intel Core Ultra / Arc Graphics (XPU).

- Automated training, validation, and checkpointing pipeline.

- Comprehensive evaluation metrics, including confusion matrices and misclassification visualizations.

- Fully automated environment setup and execution via Windows PowerShell.

## Project Structure
To run successfully, your dataset must be placed in a BrainTumorImages/ directory with subfolders for each split:
```
ProjectGreyMatter/
├── BrainTumorImages/           # Training, Validation, and Testing splits
├── brain/                      # Local Virtual Environment
├── manage.ps1                  # Windows PowerShell Automation Script
├── config.py                   # Hyperparameters & Hardware Selection
├── utils.py                    # Plotting & Helper Functions
├── ultimate_setup_dataset.py   # Downloads, Hashes, Deduplicates, Splits Dataset
├── dataset.py                  # Data Loading & Transforms
├── model.py                    # CNN Architecture
├── train.py                    # Main Entry Point
└── evaluate.py                 # Model Evaluation & Metrics
```


Each split in BrainTumorImages/ should contain subfolders representing the four classes:
```
Training/
├──glioma/
├──meningioma/
├──pituitary/
└──notumor/
```
## Model & Pipeline Architecture
Our custom baseline model follows a hierarchical feature extraction approach designed for multi-class MRI classification. It utilizes ```AdaptiveAvgPool2d``` instead of a traditional flat layer to remain robust against varying input resolutions and reduce overfitting.


### Layer Specifications

| Layer Type | Output Channels | Kernel Size | Activation |
| :--- | :--- | :--- | :--- |
| Conv Block 1 | 32 | 7×7 | ReLU + BatchNorm |
| Conv Block 2 | 64 | 3×3 | ReLU + BatchNorm |
| Conv Block 3 | 128 | 3×3 | ReLU + BatchNorm |
| Conv Block 4 | 256 | 3×3 | ReLU + BatchNorm |
| Conv Block 5 | 512 | 3×3 | ReLU + BatchNorm |
| Adaptive Pool | 512 | 1×1 | - |
| FC Layer 1 | 256 | - | ReLU + Dropout |
| Output Layer | 4 | - | Softmax (Logits) |


## Training Pipeline
- Optimizer: AdamW (Adam with Decoupled Weight Decay) for improved regularization.

- Loss Function: Cross-Entropy Loss

- Scheduler: OneCycleLR with max_lr set to 0.0003

- Hardware Acceleration: Auto-detects and utilizes the native PyTorch XPU backend (Intel Integrated/Discrete GPUs and NPUs).


## How to Run (Windows PowerShell)

The project is fully automated using manage.ps1. If you encounter a script execution error, run the following command in PowerShell first:

```Set-ExecutionPolicy RemoteSigned -Scope CurrentUser``` 


| PowerShell Command | Description|
| :--- | :--- |
| ```.\manage.ps1 all``` | Install dependencies automatically based on the hardware that is present, creates the brain venv, and downloads/creates the BrainTumorImage folders and splits: | 
| ```.\manage.ps1 run``` | Runs the full training and evaluation pipeline: |
| ```.\manage.ps1 clean``` | Clean Temporary Files Removes __pycache__, .png plots, and .pth checkpoints: |
| ```.\manage.ps1 rebuild``` | Full Rebuild Deletes the environment and reinstalls everything from scratch: |


## Evaluation & Outputs

The pipeline evaluates the model on a dedicated validation set after every epoch. It utilizes a "Best-Model" saving strategy, tracking validation accuracy and only saving the state if it outperforms previous epochs. To ensure reproducibility, the project uses a fixed transformation pipeline and a checkpoint-based testing protocol.

Generated Artifacts:
| File | Description |
| :--- | :--- |
| best_brain_tumor_cnn.pth | The highest-performing model checkpoint. |
| training_validation_curves.png | Dual-plot analyzing learning behavior (Loss & Accuracy). |
| confusion_matrix.png | Visual matrix of predicted vs. actual classifications. |
| misclassified_examples.png | Sample visualization of images the model misclassified. |

The pipeline also automatically computes Precision, Recall, F1-score, and Per-class accuracy.


## Future Improvements

While current development focuses on ResNet integration and hyperparameter tuning, future iterations of this project may explore:

- Deployment as a lightweight web application for real-time inference.

- Exploring Vision Transformers (ViT) as an alternative to convolution-based architectures.

## Authors

John Wheeler, Heriberto Rosa, Linh Luong (2025)
John Wheeler, Heriberto Rosa, Brooklyn Hunt (2026)

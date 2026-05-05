# ProjectGreyMatter II – MRI Brain Tumor Classification
## Overview

This project implements a comprehensive deep learning pipeline using PyTorch to classify grayscale MRI brain images into one of four categories: Glioma, Meningioma, Pituitary, or No Tumor. Evolving from a baseline Custom Convolutional Neural Network (CNN), the architecture now integrates ResNet50 and Vision Transformer (ViT-B/16) backbones via transfer learning.

To maximize clinical safety and accuracy, the current pipeline utilizes an advanced Ensemble Cascade Strategy, dynamic clinical class weighting (Focal Loss), and a highly customized automated preprocessing pipeline to handle varying MRI resolutions and noise.

### Key Features:

- **Advanced Preprocessing Pipeline:** Automated tight-bounding box extraction (SafeAutocropSquare) and soft-tissue contrast enhancement (CLAHE).

- **Multi-Model Support:** Train and evaluate using a custom CNN, ResNet50, or a Vision Transformer (ViT).

- **Clinical-First Ensemble Inference:** A dual-model soft-voting system featuring a "Clinical Catch" cascade to heavily penalize and override missed tumor detections.

- **Dynamic Training:** Implements AdamW optimization, OneCycleLR / ReduceLROnPlateau scheduling, and Early Stopping.

- **Explainable AI (XAI) Auditing:** Integrated Grad-CAM (for CNN/ResNet) and Attention Rollout (for Vision Transformers) to generate visual heatmaps. This ensures the models are learning genuine anatomical features rather than dataset artifacts.

- **Native Hardware Acceleration:** Optimized for Intel Core Ultra / Arc Graphics (XPU) with automated environment setup via Windows PowerShell.

## Project Structure
```
ProjectGreyMatter/
├── BrainTumorImages/           # Training, Validation, and Testing splits
├── brain/                      # Local Virtual Environment
├── manage.ps1                  # Windows PowerShell Automation Script
├── config.py                   # Hyperparameters & Hardware Selection
├── dataset.py                  # Data Loading, CLAHE, & Custom Transforms
├── ultimate_setup_dataset.py   # Downloads, Hashes, Deduplicates, Splits Dataset
├── utils.py                    # Plotting, Grad-CAM & Focal Loss Functions
├── model.py                    # CNN, ResNet50, ViT-B/16 Architectures
├── train.py                    # Main Training & Validation Entry Point
├── evaluate.py                 # Model Evaluation, Metrics, & Ensemble Inference
├── cam_utils.py                # Custom Attention Rollout & ViT reshape logic
├── visualize_model.py          # Generates XAI heatmaps for a single model's successes/failures
└── visualize_comparison.py     # Side-by-side ResNet (Grad-CAM) vs ViT (Rollout) evaluation
```

## Data Preprocessing & Augmentation
We implemented a custom robust data pipeline in ```dataset.py``` since MRI scans vary wildly in lighting, alignment, and aspect ratio:

**1. Safe Autocrop & Square (SafeAutocropSquare):**\
Uses thresholding to ignore faint background noise/text, finds the tightest bounding box around the brain tissue, and equally pads the shorter sides with black pixels to create a perfect square. This prevents the distortion of anatomical features during resizing.

**2. Contrast Limited Adaptive Histogram Equalization (ApplyCLAHE):**\
Enhances soft-tissue contrast while downplaying bright skull intensities, making tumor boundaries significantly clearer for the models.

**3. Training Augmentations:**\
To prevent overfitting, the training loader applies RandomAffine (translation and scaling), RandomHorizontalFlip, and RandomRotation.

## Model Architecture
The project supports three standalone models and one collaborative ensemble strategy.

**1. The Baseline Custom CNN**\
A hierarchical 5-block convolutional feature extractor utilizing AdaptiveAvgPool2d to remain robust against varying input resolutions. Weights are optimized via Kaiming initialization.

**2. Pre-trained ResNet50 & Vision Transformer (ViT-B/16)**\
Both models utilize ImageNet-1K pre-trained weights. Grayscale MRI tensors [B, 1, H, W] are cleanly replicated across three channels [B, 3, H, W] to leverage the learned RGB feature extractors before passing through modified fully connected classification heads.

**3. The Ensemble Cascade Strategy (Evaluation)**\
When running evaluation in ensemble mode, the script loads both ResNet50 and ViT to make collaborative predictions:

- **ViT Screener:** ViT acts as the primary screener for healthy patients. If ViT confidence in notumor exceeds a configurable threshold (default 75%), it overrides the ensemble.

- **Weighted Soft Voting:** Probabilities are blended, favoring ResNet (70%) for spatial feature extraction, but giving ViT (30%) a minority say for global dependencies.

- **The "Clinical Catch":** If ViT detects a tumor but the blended soft-voting misses it, the pipeline triggers a safety catch. It removes notumor as an option and forces the ensemble to select the most likely tumor type, minimizing dangerous false negatives.



## Training Pipeline
The training loop (```train.py```) is explicitly tuned for pseudo clinical viability:
- **Optimizer:** ```AdamW``` (Adam with Decoupled Weight Decay at 0.01) for improved regularization
- **Loss Function (Focal Loss + Clinical Priority Weights):** Cross entropy was replaced with Focal Loss ($\gamma$ = 2.0) paired with a custom clinical penalty tensor. The pipeline heavily penalizes missing highly dangerous or hard-to detect tumors
    - ```Glioma```: 2.5 weight (Highest Penalty)
    - ```Meningioma```: 2.0 weight
    - ```Pituitary```: 1.5 weight
    - ```NoTumor```: 1.0 weight (Lowest weight to encourage risk-taking tumor detection)
- **Learning Rate Schedulers:** 
    - ```OneCycleLR``` (Custom CNN)
    - ```ReduceLROnPlateau``` (ResNet & ViT - factors down by 0.1 after patience of 3)

- **Callbacks:** Model state is saved purely on Validation Accuracy. Early stopping halts training if Validation Loss fails to improve after 7 epochs. 


## How to Run (Windows PowerShell)

The project environment is fully automated using manage.ps1. If you encounter a script execution error, run the following command in PowerShell first ```Set-ExecutionPolicy RemoteSigned -Scope CurrentUser``` 


| PowerShell Command | Description|
| :--- | :--- |
| ```.\manage.ps1 all``` | Installs hardware-specific dependencies, creates the ```brain``` venv, and downloads/splits the dataset. | 
| ```.\manage.ps1 run``` | Runs the full pipeline: trains the model, evaluates metrics, and generates XAI visual heatmaps. |
| ```.\manage.ps1 clean``` | Removes ```__pycache__```, ```.png``` plots, and ```.pth``` checkpoints. |
| ```.\manage.ps1 rebuild``` | Deletes the environment and reinstalls everything from scratch: |

_(Note: To run specific steps manually, you can execute ```python train.py --model [custom_cnn|resnet|vit]```, ```python evaluate.py --model [custom_cnn|resnet|vit]```, and ```python visualize_model.py --model [custom_cnn|resnet|vit]``` directly inside the virtual environment)_

## Evaluation & Outputs

The pipeline evaluates the model on a dedicated testing set to ensure strict reproducibility. Running inference generates the following artifacts:

| File | Description |
| :--- | :--- |
| [model]_best_model.pth | The highest-performing model checkpoint. |
| [model]_training_curves.png | Dual-plot analyzing learning behavior (Loss & Accuracy) |
| [model]_confusion_matrix.png | Visual matrix of predicted vs. actual classifications. |
| [model]_misclassified_examples.png | Sample visualization of images the model misclassified. |
| [model]_explanation\_[successes/failures].png | Heat maps showing model focus (Grad-Cam/Rollout) on correct and incorrect predictions |
| comparison_[outcome].png | Side-by-Side visual comparison pitting ResNet against ViT on identical images. |
| augmented_sample_[class].png | Raw 512 x 512 augmented image samples fed to the XAI generators |

The terminal also outputs a standard ```classification_report``` detailing Precision, Recall, and F1-scores per class.


## Future Improvements

While current development has established a robust baseline for multi-model inference, future iterations of this project will focus on:

- **Ensemble Optimization:** Exploring and integrating additional model architectures (e.g., EfficientNet, ConvNeXt) to create a more diverse and highly accurate ensemble, further increasing the clinical viability of the cascade strategy.

- **Clinical Deployment:** Wrapping the trained models and ensemble logic into a lightweight web API (e.g., FastAPI or Flask) to facilitate real-time clinical testing, inference, and seamless front-end integration.


## Authors

John Wheeler, Heriberto Rosa, Linh Luong (2025)
John Wheeler, Heriberto Rosa, Brooklyn Hunt (2026)

import torch

# DEVICE CONFIGURATION
if torch.cuda.is_available():
    # NVIDIA Graphics Cards
    DEVICE = torch.device("cuda")
    print("Hardware selected: NVIDIA CUDA")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Apple Silicon (M1/M2/M3/M4) Neural Engine/GPU
    DEVICE = torch.device("mps")
    print("Hardware selected: Apple Silicon MPS")

elif hasattr(torch, "xpu") and torch.xpu.is_available():
    # Intel Core Ultra NPUs / Arc GPUs (Requires Intel Extension for PyTorch)
    DEVICE = torch.device("xpu")
    print("Hardware selected: Intel XPU (Arc/Core Ultra)")

else:
    # Fallback if no specialized hardware is found
    DEVICE = torch.device("cpu")
    print("Hardware selected: CPU")

# PATHS
TRAIN_DIR = "BrainTumorImages/Training"
VAL_DIR = "BrainTumorImages/Validation"
TEST_DIR = "BrainTumorImages/Testing"

# HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
IMAGE_SIZE = 256

# NORMALIZATION PARAMETERS FOR 1-CHANNEL (GRAYSCALE) IMAGES
MEAN = (0.5,)
STD = (0.5,)
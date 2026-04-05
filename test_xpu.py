import torch

print("1. PyTorch loaded successfully.")
print(f"2. XPU Available: {torch.xpu.is_available()}")

if torch.xpu.is_available():
    print("3. Attempting to get device name...")
    print(f"   Device: {torch.xpu.get_device_name(0)}")
    
    print("4. Attempting to push data to the NPU/GPU...")
    # This is exactly where the segfault usually happens
    x = torch.ones(5).to("xpu") 
    print("5. SUCCESS! The tensor is on the Intel hardware.")
    print(x)

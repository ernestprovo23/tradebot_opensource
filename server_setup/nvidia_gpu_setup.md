```markdown
# GPU Setup and Configuration Documentation

## Summary
This document outlines the steps taken to configure and verify GPU support for TensorFlow and PyTorch on the server. It includes environment variable settings, library installations, and verification scripts.

## System Information
- **OS**: Debian 12
- **GPUs**: 
  - NVIDIA GeForce GT 1030
  - NVIDIA GeForce RTX 3060

## Environment Setup

### Create and Activate Python Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Install Necessary Packages
```bash
pip install tensorflow torch transformers
```

### CUDA and cuDNN Installation
1. **Install CUDA Toolkit and cuDNN**:
   - Download CUDA and cuDNN from NVIDIA's website.
   - Install using dpkg and apt.

2. **Create Symbolic Links**:
   ```bash
   sudo mkdir -p /usr/local/cuda/lib64
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudart.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcufft.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcurand.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcusolver.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcusparse.so* /usr/local/cuda/lib64/
   sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so* /usr/local/cuda/lib64/
   ```

### Set Environment Variables
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Verification Scripts

### Verify TensorFlow and PyTorch GPU Availability
Create a script `verify_setup.py`:
```python
import tensorflow as tf
import torch

print("TensorFlow GPU Availability: ", len(tf.config.list_physical_devices('GPU')))
print("PyTorch CUDA available: ", torch.cuda.is_available())
print("PyTorch CUDA device count: ", torch.cuda.device_count())
```

Run the script:
```bash
python verify_setup.py
```

### Hugging Face Transformers Test Script
Create a script `test_hgf.py`:
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2', framework='pt')

result = generator("Hello, I'm using a GPU!", max_length=50, num_return_sequences=1)
print(result)
```

Run the script:
```bash
python test_hgf.py
```

## Optional Configuration

### Adjust TensorFlow's Minimum GPU Core Count Requirement
```bash
export TF_MIN_GPU_MULTIPROCESSOR_COUNT=3
```
Re-run the verification script after setting the environment variable.

## Notes
- Ensure CUDA and cuDNN versions are compatible with TensorFlow and PyTorch versions in use.
- Use `pip install --upgrade <package>` to update packages if necessary.
- Adjust environment variables and symbolic links as per future updates or changes in library paths.
```
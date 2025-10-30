# 3D Medical Image Registration

Deep learning-based 3D medical image registration using Spatial Transformer Networks (STN) for CT scan alignment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔥 **3D Spatial Transformer Network** for differentiable image registration
- 📊 **Multiple similarity metrics**: NCC, Local NCC, Mutual Information
- 📁 **Automatic experiment tracking** with checkpoints and manifests
- 🎯 **Landmark-guided augmentation** for anatomically plausible transforms
- ⚡ **Resume training** from any checkpoint
- 📈 **Comprehensive visualization** with multi-view registration results

## Quick Start

### Installation

git clone https://github.com/Reda-Abdellah/dl_registration.git
cd DL_Registration
pip install -r requirements.txt


### Data Structure

data/
├── subject_001/
│ ├── image.mha
│ └── cochlea-estimated.json # {"center": [x, y, z]}
└── subject_002/
│ ├── image.mha
│ └── cochlea-estimated.json




### Training

**Python**:
from src.train import main

results = main(
config_path='config.yaml',
experiment_name='baseline_run'
)


**CLI**:
python main.py --config config.yaml --name baseline_run


### Resume Training
**Python**:
from src.train import main

results = main(
resume_from='experiments/baseline_run/checkpoints/epoch_50.pth'
)




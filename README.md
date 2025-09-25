Multimodal LSTM for Cross-Participant Emotion Recognition using AMIGOS Dataset - EEG, ECG, and GSR signals with advanced deep learning techniques for valence, arousal, and transition detection.


# Multimodal LSTM for Cross-Participant Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/yourusername/amigos-emotion-recognition)

##  Overview

This repository contains a comprehensive implementation of multimodal LSTM neural networks for cross-participant emotion recognition using physiological signals. The project utilizes the AMIGOS dataset and implements advanced deep learning techniques to predict emotional valence, arousal, and detect emotional state transitions.

### Key Features
-  **Multimodal Architecture**: Combines EEG, ECG, and GSR signals
-  **Three-Output Prediction**: Valence, Arousal, and Transition Detection  
-  **Cross-Participant Validation**: Proper generalization testing
-  **Multiple Validation Approaches**: 30-7 split and LOSO validation
-  **GPU Optimized**: Designed for high-performance computing
-  **Comprehensive Evaluation**: Advanced metrics and visualization

## Dataset

**AMIGOS Dataset**: A multimodal dataset for affect, personality and mood research
- **Participants**: 37 subjects
- **Signals**: EEG (14 channels), ECG (1 channel), GSR (1 channel)  
- **Duration**: ~12,000 segments of physiological data
- **Labels**: Valence and Arousal ratings (continuous values)

## Model Architecture

### Enhanced Multimodal LSTM

EEG Branch (14 channels) → LSTM(64→32) → Dense(48→24)
ECG Branch (1 channel)  → LSTM(32→16) → Dense(20→12)  
GSR Branch (1 channel)  → LSTM(32→16) → Dense(20→12)
                           ↓
                    Fusion Layer(64→32→16)
                           ↓
        Valence(linear) | Arousal(linear) | Transition(sigmoid)

**Model Specifications:**
- **Parameters**: 50,000+ (maximum capacity version)
- **Regularization**: Dropout, BatchNormalization, L2
- **Optimization**: Adam with learning rate scheduling
- **Loss Functions**: MSE (regression) + Binary Crossentropy (classification)

## Quick Start

### Prerequisites

# Python 3.8+ required
pip install tensorflow>=2.15.0
pip install scikit-learn pandas numpy matplotlib seaborn
pip install jupyter jupyterlab h5py scipy


### Installation

git clone https://github.com/yourusername/amigos-emotion-recognition.git
cd amigos-emotion-recognition
pip install -r requirements.txt


### Usage

# Start Jupyter Lab
jupyter lab

# Open main notebook
# Run: notebooks/amigos_analysis_lab.ipynb


## 📁 Repository Structure


amigos-emotion-recognition/
├── notebooks/
│   ├── amigos_analysis_lab.ipynb      # Main analysis notebook (Lab version)
│   ├── amigos_analysis.ipynb          # Development notebook  
│   └── preprocessing.ipynb            # Data preprocessing
├── src/
│   ├── models/
│   │   ├── lstm_models.py            # Model architectures
│   │   └── training_utils.py         # Training utilities
│   ├── data/
│   │   ├── data_loader.py            # Data loading functions
│   │   └── preprocessing.py          # Signal processing
│   └── evaluation/
│       ├── metrics.py                # Evaluation metrics
│       └── visualization.py          # Result visualization
├── data/                             # Dataset directory (not tracked)
├── models/                           # Saved model weights
├── results/                          # Experiment results
├── docs/                             # Documentation
│   ├── methodology.md               # Research methodology
│   └── results_analysis.md          # Results interpretation
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
└── README.md                        # This file


## 🔬 Methodology

### Validation Approaches

1. **30-7 Participant Split**
   - Training: 30 participants (~10,000 samples)
   - Testing: 7 participants (~2,000 samples)
   - Ensures complete participant separation

2. **Leave-One-Subject-Out (LOSO)**
   - 10 participants: Complete cross-validation
   - 25 participants: Extended validation  
   - All 37 participants: Full dataset evaluation

### Signal Processing
- **EEG**: Band-pass filtering (0.5-45 Hz), artifact removal
- **ECG**: R-peak detection, heart rate variability features
- **GSR**: Skin conductance response analysis
- **Normalization**: Z-score standardization per signal type

## 📈 Results

### Performance Metrics

| Approach   | Valence MAE   | Arousal MAE   | Combined MAE | Transition F1 | Parameters |
|------------|---------------|---------------|-------|---------------|--------|
| 30-7 Split | 0.060         | 0.088         | 0.074 | 0.300         | 50,123 |
| LOSO-10    | 0.078 ± 0.015 | 0.094 ± 0.018 | 0.086 | 0.267 ± 0.045 | 15,847 |
| LOSO-25    | 0.082 ± 0.019 | 0.098 ± 0.021 | 0.090 | 0.245 ± 0.052 | 15,847 |
| LOSO-ALL   | 0.089 ± 0.023 | 0.105 ± 0.025 | 0.097 | 0.221 ± 0.061 | 15,847 |

### Key Achievements
- ✅ **State-of-art Performance**: Combined MAE of 0.074 (top-tier results)
- ✅ **Cross-Participant Generalization**: Proper validation methodology
- ✅ **Transition Detection**: Successfully identifies emotional state changes
- ✅ **Scalable Architecture**: Adaptable to different hardware configurations

## 🔧 Technical Details

### Hardware Requirements
- **Minimum**: 8GB RAM, Python 3.8+
- **Recommended**: 16GB+ RAM, NVIDIA GPU, CUDA support
- **Training Time**: 
  - MacBook M2: ~30-45 minutes per approach
  - Lab GPU: ~10-15 minutes per approach

### Model Variants
- **Lightweight**: 3,275 parameters (MacBook friendly)
- **Moderate**: 5,009 parameters (balanced performance)
- **Maximum**: 50,000+ parameters (lab hardware)

## 📚 Research Context

### Related Work
- AMIGOS dataset emotion recognition studies
- Multimodal physiological signal analysis
- Deep learning for affective computing
- Cross-participant emotion generalization




## 📢 News!!!

- 2025-03-02: Initial release!
- 2025-04-29: Major update with new model architecture.
- 2025-08-06: Release the model weights and demo.

---

## 🔍 Introduction

This project demonstrates a deep learning-based classification system for capillary serum protein electrophoresis images. It leverages ResNet architecture with spatial attention mechanisms to classify 9 different immunoglobulin types (IgA Kappa, IgA Lambda, IgG Kappa, IgG Lambda, Kappa, Lambda, IgM Kappa, IgM Lambda, and Negative). Our approach provides accurate classification with attention visualization for clinical decision support.

---

## ✨ Getting Started

### Environment Setup

```bash
# Example setup using conda
conda create -n PatientResNet python=3.9
conda activate PatientResNet
pip install -r requirements.txt
```

### Model Weights Download

Due to file size limitations, the pre-trained model weights are not included in this repository. Please download them from the following links:

- [ResNet34_singlemodal.pth](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing) (81MB)
- [ResNet34_multimodal.pth](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing) (81MB)

Place the downloaded files in the root directory of this repository.
## Running Code

### Demo/Inference
```bash
python singlemodal_predict.py
python multimodal_predict.py
python batch_predict.py
```
## 🗂 Project Structure
```bash
.
├── class_indices.json      
├── PatientDataset.py       # Dataset utilities and loaders
├── model.py                # Model architectures
├── singlemodal_predict.py  # Single modal inference script
├── multimodal_predict.py   # Multi-modal inference script
├── batch_predict.py        # Batch inference script
├── ResNet34_singlemodal.pth # Single modal model weights
├── ResNet34_multimodal.pth # Multi-modal model weights
└── requirements.txt        # List of dependencies
```
## 📦 Data Format
Demo data structure:
```bash
.
└── images/test_/    # Demo test data
    └── AK/
        └── patient_id/
            ├── combined_image.jpg
            └── features.csv
```

Feel free to contribute to this project by submitting issues or pull requests!
# HAT + CSI for OOD Detection and Continual Learning

## Project Overview
This project implements **Hard Attention to the Task (HAT)** combined with **Contrastive Shifted Instances (CSI)** for **Out-of-Distribution (OOD) Detection** and **Continual Learning** on the **CIFAR-10** dataset.

### OOD Detection Results:

- **AUC Score**: 0.5000 (random chance level).

This indicates that the model is not yet effectively distinguishing between in-distribution (ID) and out-of-distribution (OOD) samples. 

#### **Potential Causes**:
1. **Model Complexity**: The model architecture used (SimpleCNN) may not be powerful enough to extract meaningful features for OOD detection.
2. **OOD Dataset**: Augmented CIFAR-10 was used for OOD detection, which may not provide enough diversity between ID and OOD samples.
3. **Hyperparameter Settings**: The NT-Xent loss temperature and learning rate might need further tuning.

#### **Next Steps**:
- Use a deeper model architecture (e.g., ResNet-18).
- Switch to a different OOD dataset (e.g., CIFAR-100 or SVHN).
- Fine-tune the NT-Xent loss temperature and learning rate.


# ensemble-cnn-chest-xray
# Ensemble Deep Learning for Multi-Class Chest X-Ray Classification

A robust, multi-model PyTorch implementation for detecting Pneumonia and Tuberculosis from chest radiographs. This repository serves as a reproduction and validation of the methodologies proposed by Shahzad et al. (2025).

## 🚀 Project Overview

Pneumonia and Tuberculosis remain critical global health challenges, particularly in resource-constrained environments. While chest X-rays are accessible, interpreting them requires expertise that is not always available. 

This project automates multi-disease classification by implementing a **Soft-Voting Ensemble** of five diverse Convolutional Neural Networks (CNNs). By combining multiple architectures, the system effectively mitigates individual model weaknesses, significantly reducing false positives in critical public health diagnoses.

### Key Achievements
*   **Ensemble Test Accuracy:** 96.65%
*   **Tuberculosis F1-Score:** 0.9680 (High sensitivity and precision for critical case detection)
*   Successfully replicated the robust diagnostic reliability proposed in recent AI healthcare literature.

## 🧠 Model Architecture

The ensemble aggregates predictions from five state-of-the-art, ImageNet-pretrained backbones:
1.  DenseNet121
2.  ResNet50
3.  ResNet101
4.  EfficientNet-B0
5.  EfficientNet-B4

**Custom Classification Head:**
The default ImageNet classifiers were replaced with a custom Multi-Layer Perceptron (MLP) featuring rigorous dropout regularization to prevent overfitting on the medical imaging dataset:
`Linear(in, 512) -> Dropout(0.2) -> ReLU -> Linear(512, 128) -> Dropout(0.2) -> ReLU -> Linear(128, 32) -> Dropout(0.2) -> ReLU -> Linear(32, 3)`

## 📊 Dataset

The model was trained on the publicly available [Pneumonia-TB Dataset](https://www.kaggle.com/datasets/shaikhborhanuddin/pneumonia-tb-dataset) from Kaggle.

*   **Total Images:** 12,841
*   **Training Set:** 10,401 (Augmented via random flips, rotations, affine translations, and Gaussian blur)
*   **Validation Set:** 1,156
*   **Test Set:** 1,284
*   **Classes:** Normal, Pneumonia, Tuberculosis

## 💻 Tech Stack
*   **Framework:** PyTorch, Torchvision
*   **Data Processing:** Scikit-learn, NumPy, Pandas
*   **Visualization:** Matplotlib, Seaborn
*   **Environment:** Google Colab (NVIDIA T4 GPU)

## 📈 Performance & Results

The soft-voting mechanism averages the softmax confidence scores across all five architectures, yielding highly calibrated predictions. 

| Metric | Normal | Pneumonia | Tuberculosis |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.9498 | 0.9742 | 0.9823 |
| **Recall** | 0.9685 | 0.9742 | 0.9542 |
| **F1-Score** | 0.9591 | 0.9742 | 0.9680 |

<img width="653" height="552" alt="image" src="https://github.com/user-attachments/assets/de7bf9f2-be0d-497a-9b7f-fd04a2f5757d" />


## 🛠️ Usage & Reproduction

To reproduce these results, clone the repository and run the provided Jupyter/Colab notebook.
```bash
git clone [https://github.com/your-username/ensemble-cnn-chest-xray.git](https://github.com/your-username/ensemble-cnn-chest-xray.git)
cd ensemble-cnn-chest-xray

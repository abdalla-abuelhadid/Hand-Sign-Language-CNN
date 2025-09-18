# ✋ Hand Sign Recognition with CNN

This repository contains a Jupyter Notebook that demonstrates **American Sign Language (ASL) Hand Sign Recognition** using a **Convolutional Neural Network (CNN)**.  
The project is based on a modified version of the **Sign Language MNIST dataset**, where CSV data was converted into images (28×28 grayscale).

---

## 📊 Dataset
- **Source**: [Sign Language MNIST (Image Version)](https://www.kaggle.com/datamunge/sign-language-mnist)  
- **Total Images**: 27,455  
- **Image Format**: Grayscale JPEG  
- **Image Size**: 28×28 pixels  
- **Classes**: 24 (A–Y, excluding J and Z)  
- **Structure**: Each folder is named after the class label and contains corresponding images.  

### Data Augmentation
To improve generalization and reduce overfitting, the training pipeline applies:
- Rotation (±3°)  
- Zoom (up to 20%)  
- Horizontal flips  
- Brightness/contrast variation (±15%)  

---

## 🚀 Model
The CNN model architecture:
1. **Conv2D + MaxPooling2D**  
2. **Conv2D + MaxPooling2D**  
3. **Conv2D + MaxPooling2D**  
4. **Flatten**  
5. **Dense (512, ReLU) + L2 Regularization**  
6. **Dropout (0.5)**  
7. **Dense (24, Softmax)**  

---

## 🏋️ Training
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Metrics**: Accuracy  
- **Callbacks**: EarlyStopping (patience=5, restore best weights)  
- **Epochs**: Up to 50 (stops earlier if no improvement)  

---

## 📈 Results
- Achieved strong accuracy on both training and validation sets.  
- Visualized predictions on random validation images (correct ✅ vs incorrect ❌).  
- Training and validation accuracy/loss curves confirm reduced overfitting.  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/hand-sign-cnn.git
cd hand-sign-cnn
pip install -r requirements.txt


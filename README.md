# ✋ Hand-Sign-Language-CNN  

A **Convolutional Neural Network (CNN)** project for recognizing **American Sign Language (ASL)** hand signs from images.  
This repository provides a full workflow — from **data loading** 📂 to **model training** 🏋️ and **evaluation** 📈 — using modern deep learning practices.  

---

## 📑 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🗂️ Dataset](#%EF%B8%8F-dataset)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [▶️ Usage](#%EF%B8%8F-usage)
- [🏗️ Model Architecture](#%EF%B8%8F-model-architecture)
- [🏋️ Training](#%EF%B8%8F-training)
- [📊 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📬 Contact](#-contact)

---

## 📌 Project Overview
This project implements a **deep learning pipeline** 🧠 for classifying **ASL hand signs (A–Y, excluding J and Z)** using a CNN.  

The workflow includes:
- ✅ Data preprocessing  
- 🔄 Data augmentation  
- 🏗️ Model building  
- 🏋️ Training with Early Stopping  
- 📊 Evaluation and Visualization  

---

## 🗂️ Dataset
- **Source:** [Sign Language MNIST (Image Version)](https://www.kaggle.com/datasets/ash2703/handsignimages)  
- **Classes:** 24 (A–Y, excluding J and Z)  
- **Images:** 34,627 (Train: 27,455 | Test: 7,172)  
- **Image Size:** 28×28 pixels, grayscale  
- **Structure:** Organized in **class-named folders** 📁  
- **Preprocessing:** Cropping ✂️, resizing 📏, grayscale conversion ⚫, augmentation 🎨  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/abdalla-abuelhadid/Hand-Sign-Language-CNN.git
cd Hand-Sign-Language-CNN
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the notebook for an end-to-end demonstration:

```bash
jupyter notebook Notebook/Hand-Sign-Recognition-CNN.ipynb
```

**Main Steps in Notebook:**
1. 📥 Download and prepare the dataset (Kaggle integration).  
2. 🔄 Preprocess and augment data.  
3. 🏗️ Build and compile the CNN model.  
4. 🏋️ Train with **EarlyStopping**.  
5. 📈 Evaluate and visualize results.  
6. 💾 Save the trained model as `model.h5`.  

---

## 🏗️ Model Architecture
- **Input:** 28x28x3 (RGB, converted from grayscale)  
- **Layers:**  
  - 🔹 3× `Conv2D` + `MaxPooling2D`  
  - 🔹 `Flatten`  
  - 🔹 `Dense(512, relu, L2 regularization)`  
  - 🔹 `Dropout(0.5)`  
  - 🔹 `Dense(24, softmax)` (output)  
- **Optimizer:** ⚡ Adam  
- **Loss:** 🎯 Categorical Crossentropy  
- **Regularization:** 🛡️ L2 + Dropout  

---

## 🏋️ Training
- **Data Augmentation:** 🔄 Rotation, Zoom, Horizontal Flip  
- **EarlyStopping:** ⏹️ Monitors `val_loss` (patience=5, restores best weights)  
- **Epochs:** ⏳ Up to 50 (early stopping may cut earlier)  
- **Validation:** 📊 Performance tracked using validation split  

---

## 📊 Results
- ✅ **Test Accuracy:** ~95%  
- 🏆 **Train Accuracy:** ~98%  
- 🖼️ Visualized batch predictions (correct ✅ vs incorrect ❌)  
- 🎯 Correctly predicted **29 out of 30 samples** in testing visualization  
- 💾 Model saved as: `model.h5`  

---

## 🤝 Contributing
Contributions and suggestions are welcome!  
- Fork the repo 🍴  
- Create a new branch 🌱  
- Submit a pull request 🔄  

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

## 📬 Contact
For questions or feedback:  
👤 [Abdalla Abuelhadid](https://github.com/abdalla-abuelhadid)  

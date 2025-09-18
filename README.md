# Hand-Sign-Language-CNN

A Convolutional Neural Network (CNN) project for recognizing American Sign Language (ASL) hand signs from images. This repository provides a full workflow—from data loading to model training and evaluation—using modern deep learning practices.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project implements a deep learning pipeline for classifying ASL hand signs (A-Y, excluding J and Z) using a CNN. The workflow includes data preprocessing, augmentation, model building, training with early stopping, and evaluation.

## Dataset

- **Source:** ([Sign Language MNIST](https://www.kaggle.com/datasets/ash2703/handsignimages)), converted from CSV to JPEG images
- **Classes:** 24 (A–Y, excluding J and Z)
- **Images:** 34,627 (Train: 27,455 | Test: 7,172)
- **Image Size:** 28×28 pixels, grayscale
- **Structure:** Images organized in class-named folders
- **Preprocessing:** Cropping, resizing, grayscale conversion, augmentation (rescaling, flipping, rotation, pixelation)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/abdalla-abuelhadid/Hand-Sign-Language-CNN.git
cd Hand-Sign-Language-CNN
pip install -r requirements.txt
```

## Usage

Run the notebook for an end-to-end demonstration:

```bash
jupyter notebook Notebook/Hand-Sign-Recognition-CNN.ipynb
```

**Main Steps in Notebook:**
1. Download and prepare the dataset (Kaggle integration).
2. Preprocess and augment data.
3. Build and compile the CNN model.
4. Train the model with early stopping.
5. Evaluate and visualize results.
6. Save the trained model as `model.h5`.

## Model Architecture

- **Input:** 28x28x3 (RGB, converted from grayscale)
- **Layers:**  
  - 3× `Conv2D` + `MaxPooling2D`
  - `Flatten`
  - `Dense(512, relu, L2 regularization)`
  - `Dropout(0.5)`
  - `Dense(24, softmax)` (output)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Regularization:** L2 and Dropout

## Training

- **Augmentation:** Rotation, zoom, horizontal flip
- **EarlyStopping:** Monitors validation loss, patience=5, restores best weights
- **Epochs:** Up to 50 (with early stopping)
- **Validation:** Uses test split for performance monitoring

## Results

- **Test Accuracy:** ~95%
- **Train Accuracy:** ~98%
- **Visualization:** Batch prediction samples with correct/wrong highlight
- **Correctly predicted 29 samples out of 30 in testing phase**
- **Model Saved:** `model.h5`

## Contributing

Contributions and suggestions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, contact [abdalla-abuelhadid](https://github.com/abdalla-abuelhadid).

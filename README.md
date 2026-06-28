# Wildfire Damage Classification using Convolutional Neural Networks (CNN)

## Project Overview

This project presents a deep learning approach for classifying wildfire damage from aerial and satellite imagery using Convolutional Neural Networks (CNNs). The objective is to automatically identify different levels of structural damage after wildfire events, helping support rapid disaster assessment and recovery planning.

The project was developed as part of a Machine Learning course and demonstrates the complete workflow of an image classification system, including data preprocessing, CNN model development, training, evaluation, visualization, and performance analysis.

---

## Objectives

* Develop a CNN-based image classification model for wildfire damage assessment.
* Preprocess and prepare wildfire image datasets for deep learning.
* Train and evaluate the CNN model using TensorFlow/Keras.
* Visualize model performance through learning curves and confusion matrices.
* Compare predicted and actual classifications using standard evaluation metrics.

---

## Dataset

**Dataset Name:** Structure Wildfire Damage Classification Dataset

**Source:** Hugging Face Dataset Repository

The dataset contains wildfire-affected structural images categorized into different damage levels. Images are resized to **224 × 224** pixels before training.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Hugging Face Datasets

---

## Project Workflow

1. Load wildfire image dataset
2. Image preprocessing and resizing
3. Data normalization
4. Train-validation-test split
5. CNN model construction
6. Model training
7. Performance evaluation
8. Confusion matrix generation
9. Accuracy and loss visualization
10. Prediction and analysis

---

## CNN Architecture

The implemented CNN consists of:

* Convolutional Layers
* ReLU Activation
* MaxPooling Layers
* Dropout Layer
* Flatten Layer
* Fully Connected Dense Layer
* Softmax Output Layer

---

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report
* Training Accuracy Curve
* Validation Accuracy Curve
* Training Loss Curve
* Validation Loss Curve

---

## Repository Structure

```
Wildfire-Damage-Classification/
│
├── notebooks/
│   └── wildfire_damage_classification.ipynb
│
├── models/
│   ├── cnn_model.h5
│   └── mobilenet_model.h5
│
├── figures/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shariq7427/Wildfire-Damage-Classification.git
```

Move into the project folder:

```bash
cd Wildfire-Damage-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Results

The CNN model successfully learned meaningful image features for wildfire damage classification.

Performance was evaluated using multiple metrics including classification accuracy, confusion matrix, precision, recall, and F1-score. Training and validation curves demonstrated stable convergence with no significant overfitting under the selected training configuration.

---

## Future Improvements

* Increase dataset size
* Apply transfer learning (EfficientNet, ResNet50)
* Perform hyperparameter optimization
* Deploy the model as a web application
* Explore real-time wildfire damage assessment

---

## Author

**Shariq Ali**

Machine Learning Course Project

---

## License

This project is intended for educational and academic purposes.

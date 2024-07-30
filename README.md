# Breast Cancer Detection using Machine Learning

This project aims to build a machine learning model to detect breast cancer from histopathological images. The model is trained on a dataset of labeled images and uses various machine learning techniques to classify the images into malignant and benign categories.

## Project Overview

Breast cancer is one of the most common types of cancer among women worldwide. Early detection and diagnosis are crucial for effective treatment and improving survival rates. This project leverages machine learning algorithms to aid in the early detection of breast cancer, thereby potentially saving lives.

## Files in the Repository

- `Breast_Cancer_Detection.ipynb`: Jupyter notebook containing the data preprocessing, model training, and evaluation steps.
- `app.py`: Python script to deploy the trained model as a web application for real-time predictions.

## Key Features

- **Data Preprocessing**: Includes data loading, cleaning, normalization, and augmentation techniques to prepare the dataset for model training.
- **Model Training**: Utilizes various machine learning algorithms such as Logistic Regression, Support Vector Machines (SVM), Random Forest, and Convolutional Neural Networks (CNN).
- **Model Evaluation**: Assesses the performance of the trained models using metrics like accuracy, precision, recall, and F1-score.
- **Web Application**: A simple web application built with Flask to provide a user-friendly interface for uploading images and getting predictions.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Flask
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- OpenCV
- Matplotlib

# Sample Code
## Data Preprocessing
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('data.csv')

# Preprocess data
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Model Training
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

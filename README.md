# Tomato Leaf Disease Detection & Classification System

## Project Overview
A machine learning-based system that classifies tomato leaf images into three categories: Early Blight, Late Blight, and Healthy. The project implements multiple machine learning and deep learning algorithms and provides a REST API using Flask for image prediction.

## Dataset
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy
- Image size: 64 × 64 RGB
- Total images used: 600
- Training images: 480
- Testing images: 120

## Technologies Used
- Python
- NumPy and Matplotlib
- Scikit-learn
- PyTorch
- Flask REST API
- Pillow
- Joblib
- PCA and K-Means

## Models Implemented
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- SVM with RBF and Polynomial kernels
- Feedforward Neural Network (FNN)
- Convolutional Neural Network (CNN)
- Generative Adversarial Network (GAN)
- PCA dimensionality reduction
- K-Means clustering
- GridSearchCV hyperparameter tuning

## Model Results

| Model | Test Accuracy |
|---|---:|
| Linear SVM | 78.33% |
| Random Forest | 79.17% |
| Decision Tree | 52.50% |
| SVM with RBF Kernel | 85.00% |
| SVM with Polynomial Kernel | 79.17% |
| Tuned Random Forest | 80.83% |
| FNN | 75.00% |
| CNN | 88.33% |
| GridSearchCV-tuned SVM | 86.67% |

## REST API
The Flask API accepts a tomato leaf image and returns the predicted class in JSON format.

### Start the API
```bash
python app/app.py
```

### Test the API
In a separate terminal:
```bash
python test_api.py
```

API endpoint:
`POST /predict`

Upload the image using the form field `image`.

## Installation
```bash
python -m pip install -r requirements.txt
```

## Future Improvements
- Increase dataset size and diversity.
- Improve generalization to real-world leaf images.
- Add confidence scores to API predictions.
- Develop a web interface for uploading images.

## Note
Model accuracies depend on the dataset split and experimental configuration. The current API uses the saved GridSearchCV-tuned SVM model.

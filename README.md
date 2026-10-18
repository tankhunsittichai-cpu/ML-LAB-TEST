# Machine-Learning-Course

## LAB-2 Data Preprocessing

*   Dataset exploration
*   Data Visualization
*   Data Cleaning
*   Feature Engineering
*   References


## References
* https://www.kaggle.com/datasets/divyjain28/superstore-sales
---



## LAB-3 Regression & Classification 
**Regression**   
* Simple Linear Regression  
* Multiple Linear Regression  
* rediction

**Classification**
* Preparing Classification Data  
* Decision Boundary Visualization 
* Logistic Regression  
* Gender Prediction  
* Confusion Matrix

 **Model Comparison**
* Simple vs Multiple Linear Regression  
* Training vs Testing Performance  
* Regression vs Classification  
* Model Performance Metrics  

## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---


## LAB 4:SVM on a Dataset of Your Choice

**Objective:**
- Apply Support Vector Machine (SVM) to classify a dataset of your choice.
- Compare the performance of different SVM kernels.

**Contents:**
- Select and load a dataset of your choice.
- Explore and preprocess the dataset (clustering and classification).
- Standardize the input features before training (clustering and classification).
- Train SVM models using three kernels: (Linear, Polynomial, RBF).
- Evaluate each model using accuracy.

**Output:**
- Accuracy scores for each SVM kernel.
- The best kernel based on the test accuracy.
- A brief discussion of the experimental results.
## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---

 ## LAB-5 Support Vector Machine (SVM)

**Data Preparation**
* Dataset Selection & Loading
* Exploring & Preprocessing Data
* Feature Standardization

**SVM Classification**
* Support Vector Machine (SVM) Implementation
* Linear Kernel Training
* Polynomial Kernel Training
* RBF Kernel Training
* Ethnicity Prediction

**Model Comparison & Evaluation**
* Kernel Performance Comparison (Linear vs Polynomial vs RBF)
* Accuracy Score Evaluation
* Model Predictions Output

## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---
 
## LAB-6 Neural Network (NN)

**Data Preparation**
* Dataset Selection & Loading[cite: 24]
* Exploring & Preprocessing Data[cite: 24]
* Feature Standardization[cite: 24]

**NN Classification**
* Neural Network (NN) Implementation[cite: 24]
* Model 1 Training (e.g., 20 Epochs)[cite: 24]
* Model 2 Training (e.g., 50 Epochs)[cite: 24]
* Ethnicity Prediction[cite: 24]

**Model Comparison & Evaluation**
* Configuration Performance Comparison (Hidden Layers & Neurons)[cite: 24]
* Accuracy & Loss Score Evaluation[cite: 24]
* Model Predictions Output[cite: 24]

## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---

## LAB-7 Convolutional Neural Network (CNN)

**Data Preparation**
* Dataset Selection & Loading[cite: 30]
* Exploring & Preprocessing Data[cite: 30]
* Feature Standardization[cite: 30]

**CNN Classification**
* Convolutional Neural Network (CNN) Implementation[cite: 30]
* Model 1 Training (20 Epochs)[cite: 30]
* Model 2 Training (50 Epochs)[cite: 30]
* Ethnicity Prediction[cite: 30]

**Model Comparison & Evaluation**
* Configuration Performance Comparison (Model 1 vs Model 2)[cite: 30]
* Accuracy & Loss Score Evaluation[cite: 30]
* Model Predictions Output[cite: 30]

 ## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---

# LAB 8 - Deep Convolutional Neural Network (DCNN) Classification

This project implements a Deep Convolutional Neural Network (DCNN) to classify images from a selected dataset. It focuses on evaluating and comparing model performance across different epoch counts, hyperparameter configurations, and layer architectures.

---

## 1. Project Structure

* `main.py` : The main execution pipeline.
* `data_loader.py` : Loads the selected dataset.
* `preprocessing.py` : Cleans data, normalizes/standardizes features, and reshapes data into 2D image formats for DCNN.
* `split_data.py` : Splits data into Training and Testing sets.
* `dcnn_model.py` : Builds and trains various Deep CNN model configurations.
* `evaluate.py` : Evaluates model accuracy, generates loss/accuracy curves, and outputs confusion matrices.
* `test_dcnn.py` : Script for testing individual predictions using the trained models.
* `outputs/` : Directory for saved models (`.keras` / `.h5`), scalers, evaluation graphs (`.png`), and execution logs.
* `dataset/` : The raw dataset directory (e.g., image files or CSV format).

---

## 2. Objectives

1. Apply a Deep Convolutional Neural Network (DCNN) to classify a dataset of your choice.
2. Compare model performance using different numbers of training epochs.
3. Compare different DCNN configurations (e.g., varying the number of convolutional layers, filters, and neurons).
4. Evaluate and analyze each configuration using accuracy, loss curves, and prediction results.

---

## 3. Workflow

```text
Data Loading ➔ Preprocessing & Standardization ➔ Train/Test Split ➔ DCNN Model Building (Multiple Configs) ➔ Model Training & Epoch Tuning ➔ Evaluation & Results Export


 ## References
https://www.kaggle.com/code/shahraizanwar/age-gender-ethnicity-prediction
 ---

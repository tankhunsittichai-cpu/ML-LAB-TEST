# LAB 7 - Convolutional Neural Network Classification

This project implements a Convolutional Neural Network (CNN) to classify facial images by ethnicity. It focuses on comparing different model configurations, including convolutional layers, neurons, and training epochs.

### 1. Project Structure
*   **`main.py`**: The main execution pipeline.
*   **`data_loader.py`**: Loads the CSV dataset.
*   **`preprocessing.py`**: Cleans data, converts pixel strings to arrays, applies Label Encoding, and standardizes/reshapes features into 2D image formats for CNN.
*   **`split_data.py`**: Splits data into Training (70%) and Testing (30%) sets.
*   **`cnn_model.py`**: Builds and trains the TensorFlow/Keras CNN models.
*   **`evaluate.py`**: Evaluates accuracy and generates loss/accuracy graphs, as well as the confusion matrix.
*   **`test_cnn.py`**: Script for testing individual predictions using the saved model.
*   **`outputs/`**: Directory for saved models (`.keras`), scalers (`.pkl`), graphs (`.png`), and evaluation logs.
*   **`age_gender(1)_2.csv`**: The raw dataset (must be placed in the root folder).

### 2. Dataset
*   **Target:** `ethnicity` (encoded using Label Encoding)
*   **Features:** `pixels` (converted from strings to numeric arrays and reshaped to 2D image matrices)

### 3. Objectives
1. Apply a Convolutional Neural Network (CNN) for image classification.
2. Compare model performance across different training epochs (e.g., 20 vs. 50).
3. Compare different CNN architectures (e.g., 1 Conv layer vs. 2 Conv layers with Dropout).

### 4. Workflow
Data Loading ➔ Preprocessing (Encoding, Standardization & Reshaping) ➔ Train/Test Split ➔ Model Training (2 configurations) ➔ Evaluation & Export (Graphs/Confusion Matrix/Models).

### 5. Key Techniques
*   **Convolutional Neural Network (CNN):** Built using TensorFlow/Keras (Conv2D, MaxPooling2D, Flatten).
*   **Activation Functions:** ReLU (Convolutional/Hidden Layers) and Softmax (Output Layer for multi-class classification).
*   **Regularization:** Dropout layer implemented to prevent model overfitting during extended training.
*   **Optimizer:** Adam Optimizer.

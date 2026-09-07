# LAB 6 - Neural Network Classification

This project implements a Neural Network (Multi-Layer Perceptron) to classify facial images by ethnicity. It focuses on comparing different model configurations, including hidden layers, neurons, and training epochs.

### 1. Project Structure
*   **`main.py`**: The main execution pipeline.
*   **`data_loader.py`**: Loads the CSV dataset.
*   **`preprocess.py`**: Cleans data, converts pixel strings to arrays, applies Label Encoding, and standardizes features.
*   **`spilt_data.py`**: Splits data into Training (70%) and Testing (30%) sets.
*   **`nn_model.py`**: Builds and trains the TensorFlow/Keras models.
*   **`evaluate.py`**: Evaluates accuracy and generates loss/accuracy graphs.
*   **`test_nn.py`**: Script for testing individual predictions.
*   **`output/`**: Directory for saved graphs (`.png`) and evaluation logs (`.txt`).
*   **`age_gender(1)_2.csv`**: The raw dataset (must be placed in the root folder).

### 2. Dataset
*   **Target:** `ethnicity`
*   **Features:** `pixels` (converted from strings to numeric arrays)

### 3. Objectives
1. Apply a Neural Network for image classification.
2. Compare model performance across different training epochs (e.g., 20 vs. 50).
3. Compare different NN architectures (e.g., 1 hidden layer with 32 neurons vs. 2 hidden layers with 64-32 neurons).

### 4. Workflow
Data Loading ➔ Preprocessing (Encoding & Standardization) ➔ Train/Test Split ➔ Model Training (3 configurations) ➔ Evaluation & Export (Graphs/Logs).

### 5. Key Techniques
*   **Neural Network (MLP):** Built using TensorFlow/Keras.
*   **Activation Functions:** ReLU (Hidden Layers) and Softmax (Output Layer for multi-class classification).
*   **Optimizer:** Adam Optimizer.
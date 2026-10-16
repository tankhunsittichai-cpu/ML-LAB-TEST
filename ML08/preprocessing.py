def preprocess_data(X_train, X_test):
    # Standardize / Normalize input features ให้อยู่ในช่วง [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test
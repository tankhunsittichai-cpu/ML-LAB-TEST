from data_load import load_data
from preprocess import preprocess_data, standardize_features
from svm_model import train_svm

def run_test():
    # ทดสอบการทำงานของฟังก์ชันแบบสั้นๆ
    df = load_data('age_gender(1).csv')
    X, y = preprocess_data(df)
    
    # ปรับสเกลทั้งหมด
    X_scaled, _ = standardize_features(X, X)
    
    # ใช้ RBF เป็นตัวอย่างทดสอบ
    model = train_svm(X_scaled, y, kernel_type='rbf')
    
    # ทดสอบทำนายรูปแรกใน Dataset
    sample = X_scaled[0].reshape(1, -1)
    prediction = model.predict(sample)
    
    print("--- Single Prediction Test ---")
    print(f"Actual Class: {y[0]}")
    print(f"Predicted Class: {prediction[0]}")

if __name__ == "__main__":
    run_test()
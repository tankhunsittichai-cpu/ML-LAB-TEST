from data_loader import load_data
from preprocessing import preprocess_data, standardize_features
from nn_model import build_nn
import numpy as np

def run_test():
    # โหลดและเตรียมข้อมูล
    df = load_data('age_gender(1)_2.csv')
    X, y, le = preprocess_data(df)
    X_scaled, _ = standardize_features(X, X)
    
    # สร้างโมเดลแบบ 1 เลเยอร์
    input_dim = X_scaled.shape[1]
    num_classes = len(le.classes_)
    model = build_nn(input_dim, num_classes, hidden_layers=[32])
    
    # เทรนหลอกๆ 1 รอบเพื่อให้โมเดลพร้อมทำนาย
    model.fit(X_scaled[:10], y[:10], epochs=1, verbose=0)
    
    # ทำนายภาพแรก
    sample = X_scaled[0].reshape(1, -1)
    pred_probs = model.predict(sample, verbose=0)
    predicted_class = np.argmax(pred_probs, axis=1)[0]
    
    print("--- Single Prediction Test ---")
    print(f"Actual Class (Encoded): {y[0]}")
    print(f"Predicted Class (Encoded): {predicted_class}")

if __name__ == "__main__":
    run_test()
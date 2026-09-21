import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_loader import load_data
from preprocessing import extract_features_targets
import os

def run_test():
    print("Loading test environment...")
    output_dir = 'outputs'
    model_path = os.path.join(output_dir, 'cnn_model.keras')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ ไม่พบไฟล์ Model หรือ Scaler กรุณารันไฟล์ main.py ให้เสร็จก่อนครับ")
        return

    # โหลดโมเดลและตัวปรับสเกลกลับขึ้นมา
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # โหลดข้อมูลตัวอย่าง
    df = load_data('age_gender(1)_2.csv')
    X, y, img_size, _ = extract_features_targets(df)
    
    # จำลองการดึงรูปที่ 0 มาเทสทำนายผล
    sample_idx = 0
    sample_img = X[sample_idx].reshape(1, -1)
    actual_class = y[sample_idx]
    
    # ปรับสเกลด้วย Scaler ตัวเดิม และ Reshape
    sample_scaled = scaler.transform(sample_img)
    sample_cnn = sample_scaled.reshape(1, img_size, img_size, 1)
    
    # สั่งให้โมเดลทำนายผล
    prediction_probs = model.predict(sample_cnn, verbose=0)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]
    
    print("\n--- Single Image Prediction Test ---")
    print(f"Actual Class: {actual_class}")
    print(f"Predicted Class: {predicted_class}")
    print("------------------------------------")
    if actual_class == predicted_class:
        print("✅ โมเดลทายถูก!")
    else:
        print("❌ โมเดลทายผิด!")

if __name__ == "__main__":
    run_test()
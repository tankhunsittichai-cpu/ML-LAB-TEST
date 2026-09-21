import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def extract_features_targets(df):
    """แปลงคอลัมน์ pixels เป็น numpy array และดึง target"""
    X = np.array(df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float32)).tolist())
    
    # เพิ่ม LabelEncoder ตรงนี้ เพื่อแปลงคลาสให้อยู่ในรูปแบบ 0, 1, 2, 3... เสมอ
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['ethnicity'].values)
    
    # คำนวณขนาดความกว้าง/ยาวของภาพ (เช่น 2304 พิกเซล = 48x48)
    img_size = int(np.sqrt(X.shape[1]))
    num_classes = len(np.unique(y))
    
    return X, y, img_size, num_classes

def standardize_and_reshape(X_train, X_test, img_size, output_dir):
    """ปรับสเกลข้อมูลและ Reshape ข้อมูลให้เข้ากับ CNN"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # บันทึก Scaler 
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Reshape เป็น (จำนวนข้อมูล, กว้าง, ยาว, Channels=1 สำหรับภาพ Grayscale)
    X_train_cnn = X_train_scaled.reshape(-1, img_size, img_size, 1)
    X_test_cnn = X_test_scaled.reshape(-1, img_size, img_size, 1)
    
    return X_train_cnn, X_test_cnn
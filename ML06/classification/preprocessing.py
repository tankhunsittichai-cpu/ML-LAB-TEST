import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    แปลงข้อมูลคอลัมน์ pixels เป็น numpy array และเลือก Target
    * ใช้ทาย ethnicity เหมือน Lab ก่อนหน้า เพื่อให้มีคลาสในการทายผล
    """
    X = np.array(df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float32)).tolist())
    y = df['ethnicity'].values
    
    # แปลง Target ให้เริ่มที่ 0 (Label Encoding) เพื่อให้เข้ากับโมเดล Neural Network
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def standardize_features(X_train, X_test):
    """ปรับสเกลข้อมูล (Standardize) ให้อยู่ในมาตรฐานเดียวกัน"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    แปลงข้อมูลคอลัมน์ pixels เป็น numpy array
    """
    X = np.array(df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float32)).tolist())
    
    # เปลี่ยนจาก df['gender'] เป็น df['ethnicity'] ตรงนี้ครับ <<<
    y = df['ethnicity'].values
    
    return X, y

def standardize_features(X_train, X_test):
    """ปรับสเกลข้อมูล (Standardize) ให้อยู่ในมาตรฐานเดียวกัน"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
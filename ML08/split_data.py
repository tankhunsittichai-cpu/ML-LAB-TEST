from sklearn.model_selection import train_test_split

def split_validation_set(X_train, y_train, val_size=0.2):
    # แบ่งข้อมูล Train ออกเป็น Train และ Validation เพิ่มเติมตามต้องการ
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    return X_tr, X_val, y_tr, y_val
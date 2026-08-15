from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel_type='linear'):
    """สร้างและเทรนโมเดล SVM ตาม Kernel ที่ระบุ"""
    model = SVC(kernel=kernel_type, random_state=42)
    model.fit(X_train, y_train)
    return model
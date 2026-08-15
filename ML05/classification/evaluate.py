from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """วัดผลโมเดลด้วย Accuracy และคืนค่าผลการทำนาย"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions
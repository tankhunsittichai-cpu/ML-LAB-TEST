import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, X_test, y_test):
    """ประเมินความแม่นยำ (Accuracy) บน Test Set"""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # ทำนายผล
    y_pred_probs = model.predict(X_test, verbose=0)
    predictions = np.argmax(y_pred_probs, axis=1)
    
    return accuracy, predictions

def plot_history(history, title, output_dir='output'):
    """พล็อตกราฟ Training & Validation Loss / Accuracy และเซฟเป็นรูปภาพ"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # กราฟ Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # กราฟ Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    # เซฟกราฟลงโฟลเดอร์
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()
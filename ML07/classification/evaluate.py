import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_history(history, title, filename, output_dir):
    """พล็อตและเซฟกราฟ Accuracy / Loss แยกแต่ละโมเดล"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{title} - Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{title} - Loss')
    ax2.legend()
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, output_dir):
    """พล็อตและเซฟกราฟ Confusion Matrix (cm_cnn)"""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (CNN)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'cm_cnn.png'))
    plt.close()

def plot_research_paper_graphs(hist1, hist2, output_dir):
    """พล็อตและเซฟกราฟเปรียบเทียบสไตล์ Research Paper 4 ช่อง"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CNN Configuration Comparison (Research Paper Style)')
    
    # ข้อมูลจาก Model 1 (20 Epochs)
    axs[0, 0].plot(hist1.history['accuracy'], label='Train Acc', color='blue')
    axs[0, 0].plot(hist1.history['val_accuracy'], label='Val Acc', color='lightblue')
    axs[0, 0].set_title('Model 1 (20 Epochs) - Accuracy')
    
    axs[0, 1].plot(hist1.history['loss'], label='Train Loss', color='red')
    axs[0, 1].plot(hist1.history['val_loss'], label='Val Loss', color='salmon')
    axs[0, 1].set_title('Model 1 (20 Epochs) - Loss')
    
    # ข้อมูลจาก Model 2 (50 Epochs)
    axs[1, 0].plot(hist2.history['accuracy'], label='Train Acc', color='blue')
    axs[1, 0].plot(hist2.history['val_accuracy'], label='Val Acc', color='lightblue')
    axs[1, 0].set_title('Model 2 (50 Epochs) - Accuracy')
    
    axs[1, 1].plot(hist2.history['loss'], label='Train Loss', color='red')
    axs[1, 1].plot(hist2.history['val_loss'], label='Val Loss', color='salmon')
    axs[1, 1].set_title('Model 2 (50 Epochs) - Loss')
    
    for ax in axs.flat:
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'research_paper_graphs.png'))
    plt.close()
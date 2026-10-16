import os
import matplotlib.pyplot as plt
from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_validation_set
from vgg_model import create_dcnn_model

def main():
    print("--- 1. กำลังโหลดชุดข้อมูล MNIST ---")
    (X_train, y_train), (X_test, y_test) = load_data()
    
    print("--- 2. กำลังเตรียมข้อมูล (Preprocess / Standardize) ---")
    X_train, X_test = preprocess_data(X_train, X_test)
    
    print("--- 3. แบ่งข้อมูล Train / Validation Set ---")
    X_tr, X_val, y_tr, y_val = split_validation_set(X_train, y_train, val_size=0.2)
    
    epochs_list = [5, 10]
    num_conv_layers_list = [2]
    
    histories = {}
    
    for epochs in epochs_list:
        for num_layers in num_conv_layers_list:
            print(f"\n==========================================")
            print(f" TRAINING: Epochs = {epochs}, Conv Layers = {num_layers}")
            print(f"==========================================")
            
            model = create_dcnn_model(input_shape=(28, 28, 1), num_classes=10, num_conv_layers=num_layers)
            
            history = model.fit(
                X_tr, y_tr, 
                epochs=epochs, 
                batch_size=64,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            print(f"\n--- Evaluation Output for {epochs} Epochs ---")
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"-> Test Accuracy: {test_acc * 100:.2f}%")
            print(f"-> Test Loss: {test_loss:.4f}")
            
            key = f"Epochs_{epochs}_Layers_{num_layers}"
            histories[key] = history

    print("\n--- 4. กำลังสร้างและบันทึกกราฟลง Folder output ---")
    
    # ตรวจสอบและสร้างโฟลเดอร์ชื่อ 'output' ถ้ายังไม่มี
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 5))
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    for key, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{key} (Train)')
        plt.plot(history.history['val_accuracy'], linestyle='--', label=f'{key} (Val)')
    plt.title('DCNN Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # กราฟ Loss
    plt.subplot(1, 2, 2)
    for key, history in histories.items():
        plt.plot(history.history['loss'], label=f'{key} (Train)')
        plt.plot(history.history['val_loss'], linestyle='--', label=f'{key} (Val)')
    plt.title('DCNN Model Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # บันทึกภาพกราฟลงในโฟลเดอร์ output
    output_path = os.path.join(output_dir, "dcnn_comparison_result.png")
    plt.savefig(output_path, dpi=300)
    print(f"บันทึกไฟล์กราฟเรียบร้อยแล้วที่: {output_path}")
    
    # แสดงภาพกราฟขึ้นมาดูด้วย
    plt.show()

if __name__ == "__main__":
    main()
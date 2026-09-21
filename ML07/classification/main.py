import os
from data_loader import load_data
from preprocessing import extract_features_targets, standardize_and_reshape
from split_data import split_dataset
from cnn_model import build_model_1, build_model_2
from evaluate import plot_history, plot_confusion_matrix, plot_research_paper_graphs

def main():
    # 0. จัดการโฟลเดอร์สำหรับผลลัพธ์
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    filepath = 'age_gender(1).csv'
    
    # 1. โหลดข้อมูล
    print("1. Loading dataset...")
    df = load_data(filepath)
    
    # 2. แกะ Features และ Targets
    print("2. Extracting features and targets...")
    X, y, img_size, num_classes = extract_features_targets(df)
    
    # 3. แบ่งชุดข้อมูล Train/Test
    print("3. Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 4. Standardize และ Reshape เป็น 2D Image ให้ CNN
    print("4. Standardizing and reshaping for CNN...")
    X_train_cnn, X_test_cnn = standardize_and_reshape(X_train, X_test, img_size, output_dir)
    input_shape = (img_size, img_size, 1)
    
    # 5. เทรน Model 1 (20 Epochs)
    print("\n--- 5. Training Model 1 (20 Epochs) ---")
    model_1 = build_model_1(input_shape, num_classes)
    hist_1 = model_1.fit(X_train_cnn, y_train, epochs=20, validation_data=(X_test_cnn, y_test), verbose=1)
    plot_history(hist_1, "Model 1", "history_Model_1_20Epochs.png", output_dir)
    
    # 6. เทรน Model 2 (50 Epochs)
    print("\n--- 6. Training Model 2 (50 Epochs) ---")
    model_2 = build_model_2(input_shape, num_classes)
    hist_2 = model_2.fit(X_train_cnn, y_train, epochs=50, validation_data=(X_test_cnn, y_test), verbose=1)
    plot_history(hist_2, "Model 2", "history_Model_2_50Epochs.png", output_dir)
    
    # 7. สร้างกราฟและประเมินผลรวม
    print("\n7. Generating Evaluation Graphs & Confusion Matrix...")
    plot_research_paper_graphs(hist_1, hist_2, output_dir)
    plot_confusion_matrix(model_2, X_test_cnn, y_test, output_dir)
    
    # 8. บันทึกโมเดลไว้ใช้งานต่อ
    print("8. Saving the trained model (.keras)...")
    model_2.save(os.path.join(output_dir, 'cnn_model.keras'))
    
    print(f"\n✅ ประมวลผลเสร็จสิ้น! ดูไฟล์ผลลัพธ์ทั้งหมดได้ที่โฟลเดอร์: {output_dir}")

if __name__ == "__main__":
    main()
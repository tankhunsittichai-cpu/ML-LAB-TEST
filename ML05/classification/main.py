import os
from data_load import load_data
from preprocess import preprocess_data, standardize_features
from split_data import split_dataset
from svm_model import train_svm
from evaluate import evaluate_model

def main():
    # 1. โหลดข้อมูล
    filepath = 'age_gender(1).csv'
    print("1. Loading dataset...")
    df = load_data(filepath)
    
    # 2. เตรียมข้อมูล (ทาย ethnicity)
    print("2. Preprocessing dataset...")
    X, y = preprocess_data(df)
    
    # 3. แบ่งชุดข้อมูลเป็น Train/Test
    print("3. Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 4. Standardize ข้อมูล
    print("4. Standardizing input features...")
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    # --- ส่วนจัดการโฟลเดอร์ Output ---
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True) # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    output_file = os.path.join(output_dir, 'evaluation_results.txt')
    
    # 5. เทรนโมเดลและบันทึกผล
    kernels = ['linear', 'poly', 'rbf']
    
    print("\n================ EVALUATION OUTPUT ================")
    
    # เปิดไฟล์เพื่อเขียนข้อมูลลงไป
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("================ SVM EVALUATION RESULTS ================\n\n")
        
        for kernel in kernels:
            # เทรนโมเดล
            model = train_svm(X_train_scaled, y_train, kernel_type=kernel)
            
            # ประเมินผล
            accuracy, predictions = evaluate_model(model, X_test_scaled, y_test)
            
            # เตรียมข้อความผลลัพธ์
            result_text = (
                f"[ SVM Kernel: {kernel.upper()} ]\n"
                f"Accuracy Score: {accuracy * 100:.2f}%\n"
                f"Predictions:    {predictions}\n"
                f"Actual Targets: {y_test}\n"
                f"{'-' * 50}\n"
            )
            
            # ปริ้นโชว์บนหน้าจอ (Terminal)
            print(result_text)
            
            # เขียนลงไฟล์ txt
            f.write(result_text + "\n")
            
    print(f"✅ รันเสร็จสิ้น! บันทึกผลลัพธ์เรียบร้อยแล้วที่ไฟล์: {output_file}")

if __name__ == "__main__":
    main()
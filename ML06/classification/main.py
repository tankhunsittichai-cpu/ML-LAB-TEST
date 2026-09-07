import os
from data_loader import load_data
from preprocessing import preprocess_data, standardize_features
from spilt_data import split_dataset
from nn_model import build_nn, train_model
from evaluate import evaluate_model, plot_history

def main():
    filepath = 'age_gender(1).csv'
    print("1. Loading dataset...")
    df = load_data(filepath)
    
    print("2. Preprocessing & Standardizing...")
    X, y, label_encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    input_dim = X_train_scaled.shape[1]
    num_classes = len(label_encoder.classes_)
    
    # -----------------------------------------------------
    # เปรียบเทียบ Configurations (Layers, Neurons, Epochs) ตามโจทย์
    # -----------------------------------------------------
    configs = [
        {"name": "Config 1 (1 Hidden, 32 Neurons, 20 Epochs)", "layers": [32], "epochs": 20},
        {"name": "Config 2 (2 Hidden, 64-32 Neurons, 20 Epochs)", "layers": [64, 32], "epochs": 20},
        {"name": "Config 3 (2 Hidden, 64-32 Neurons, 50 Epochs)", "layers": [64, 32], "epochs": 50}
    ]
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'nn_evaluation_results.txt')
    
    print("\n================ NN EVALUATION OUTPUT ================")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("================ NEURAL NETWORK EVALUATION RESULTS ================\n\n")
        
        for config in configs:
            name = config["name"]
            layers = config["layers"]
            epochs = config["epochs"]
            print(f"Training: {name} ... (Please wait)")
            
            # สร้างและฝึกสอนโมเดล
            model = build_nn(input_dim, num_classes, hidden_layers=layers)
            history = train_model(model, X_train_scaled, y_train, epochs=epochs)
            
            # ประเมินผล
            accuracy, predictions = evaluate_model(model, X_test_scaled, y_test)
            
            # พล็อตกราฟ Accuracy/Loss
            plot_history(history, title=name, output_dir=output_dir)
            
            # สรุปผลลัพธ์
            result_text = (
                f"[{name}]\n"
                f"- Hidden Layers Config: {layers}\n"
                f"- Epochs Run: {epochs}\n"
                f"- Test Accuracy: {accuracy * 100:.2f}%\n"
                f"- Sample Predictions: {predictions[:10]}\n"
                f"- Sample Actuals:     {y_test[:10]}\n"
                f"{'-' * 60}\n"
            )
            print(result_text)
            f.write(result_text + "\n")
            
    print(f"✅ รันสำเร็จ! เซฟกราฟประเมินผลและการเปรียบเทียบโมเดลไว้ในโฟลเดอร์ '{output_dir}'")

if __name__ == "__main__":
    main()
import os
from data_loader import load_data
from DCNN_model import build_dcnn_model
from evaluate import evaluate_model, plot_training_history
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# กำหนดเส้นทางไปยังโฟลเดอร์ Dataset (ปรับเปลี่ยนตามตำแหน่งที่คุณเก็บบนเครื่อง)
DATASET_DIR = os.path.join(os.getcwd(), 'dataset', 'chest_xray', 'chest_xray')
print("Path to dataset files:", DATASET_DIR)

# 1. โหลดข้อมูล
train_gen, val_gen, test_gen = load_data(DATASET_DIR)

# 2. สร้างโมเดล
model = build_dcnn_model(input_shape=(224, 224, 1))
model.summary()


# คำนวณน้ำหนักความสมดุลของข้อมูลระหว่าง Normal และ Pneumonia อัตโนมัติจากชุดเทรน
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights ที่ใช้:", class_weight_dict)
# 3. เทรนโมเดล
EPOCHS = 15
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict
)
model.save('pneumonia_model.keras')
print("บันทึกโมเดลเรียบร้อยแล้ว!")

# 4. พล็อตกราฟผลการเทรน
plot_training_history(history)

# 5. ประเมินผลลัพธ์บน Test Set
evaluate_model(model, test_gen)
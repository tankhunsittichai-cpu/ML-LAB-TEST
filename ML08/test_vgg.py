import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_vgg_style_model(input_shape=(32, 32, 3), num_classes=10):
    """
    สร้างโมเดลจำลองโครงสร้างแบบ VGG (VGG-like DCNN) สำหรับชุดข้อมูลภาพขนาดเล็ก (เช่น CIFAR-10)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Fully Connected Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("--- เริ่มต้นโหลดและเตรียมข้อมูลสำหรับ Test VGG ---")
    
    # 1. โหลดข้อมูลตัวอย่าง (เช่น CIFAR-10)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # 2. Standardize / Normalize ข้อมูลให้อยู่ในช่วง [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # 3. สร้างโมเดล VGG
    model = build_vgg_style_model()
    model.summary()
    
    print("\n--- กำลังฝึกสอน (Training) โมเดล VGG ---")
    # 4. เทรนโมเดล (สามารถปรับจำนวน epochs ตามต้องการ เช่น 5 หรือ 10)
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=64, 
        validation_split=0.2
    )
    
    print("\n--- ประเมินผลโมเดล (Evaluation) ---")
    # 5. ประเมินผลกับชุดทดสอบ Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # 6. บันทึกโมเดล (ถ้าต้องการ)
    model.save('vgg_cifar10_model.h5')
    print("บันทึกโมเดลเรียบร้อยแล้วในชื่อ 'vgg_cifar10_model.h5'")

if __name__ == '__main__':
    main()
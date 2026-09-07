from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_nn(input_dim, num_classes, hidden_layers=[64, 32]):
    """
    สร้างโมเดล Neural Network
    :param hidden_layers: List ระบุจำนวน Neurons ในแต่ละ Hidden Layer
    """
    model = Sequential()
    
    # Input layer + Hidden layer ตัวแรก
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    
    # สร้าง Hidden layers เพิ่มเติมตามที่กำหนดใน List
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation='relu'))
        
    # Output layer (ใช้ Softmax สำหรับการแยกหลายคลาส)
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=50, validation_split=0.2):
    """เทรนโมเดล และคืนค่าประวัติ (History) สำหรับการพล็อตกราฟ"""
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        validation_split=validation_split, 
                        verbose=0) # verbose=0 เพื่อไม่ให้รกหน้าจอตอนรัน
    return history
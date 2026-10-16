from tensorflow.keras import layers, models

def create_dcnn_model(input_shape=(28, 28, 1), num_classes=10, num_conv_layers=2):
    model = models.Sequential()
    
    # เพิ่ม Convolutional Layers ตามการกำหนดค่า (Configuration)
    for i in range(num_conv_layers):
        filters = 32 * (2**i)  # เพิ่มจำนวน filter ตามความลึก
        if i == 0:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
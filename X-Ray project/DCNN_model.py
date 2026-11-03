import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_dcnn_model(input_shape=(224, 224, 1)):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005), input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Block 4 (กำหนดชื่อไว้สำหรับทำ Grad-CAM แสดงจุดโรค)
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005), name='target_conv_layer'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Global Average Pooling แทน Flatten เพื่อลด Overfitting
        GlobalAveragePooling2D(),
        
        # Fully Connected Layers
        Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary Classification (Normal vs Pneumonia)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model
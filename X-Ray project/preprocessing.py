import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_generators(train_dir, val_dir, test_dir):
    # ทำ Data Augmentation สำหรับชุดข้อมูล Train เพื่อป้องกัน Overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False # ภาพ X-ray ปอดไม่ควรพลิกซ้าย-ขวาเนื่องจากตำแหน่งอวัยวะจะเปลี่ยน
    )

    # ชุดข้อมูล Validation และ Test จะทำแค่การ Rescale เท่านั้น
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False # ห้าม shuffle ชุด Test เพื่อให้ตรงกับ Label จริงตอนประเมินผล
    )

    return train_generator, val_generator, test_generator
import tensorflow as tf

def load_data():
    # โหลดชุดข้อมูล MNIST (โหลดไวมาก ขนาดเล็ก)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # ขยายมิติภาพให้เป็นรูปแบบของ DCNN (ความกว้าง, ความสูง, แชนแนลสี 1 ช่อง)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return (X_train, y_train), (X_test, y_test)
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)
CORS(app) # อนุญาตให้เว็บเรียกใช้งานได้

# โหลดโมเดลที่เพื่อนเทรนไว้ (เช็คชื่อไฟล์โมเดลให้ตรงกับเครื่อง เช่น pneumonia_model.keras)
model = tf.keras.models.load_model('pneumonia_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_path = 'temp.jpg'
    file.save(img_path)
    
    # ปรับขนาดรูปให้ตรงกับที่โมเดลเทรน (224x224, Grayscale)
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 
    
    # ทำนายผล
    prediction = model.predict(x)
    score = float(prediction[0][0]) 
    
    result_label = "Pneumonia" if score > 0.5 else "Normal"
    confidence = score * 100 if score > 0.5 else (1 - score) * 100

    return jsonify({
        'prediction': result_label,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
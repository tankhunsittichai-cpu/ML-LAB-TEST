from ultralytics import YOLO

def main():
    # โหลดโมเดลเริ่มต้น YOLOv8 Nano
    model = YOLO("yolov8n.pt")
    
    # เริ่มเทรนโมเดลด้วย dataset ใหม่ 3 คลาส
    results = model.train(
        data="data.yaml",               # ไฟล์ตั้งค่าข้อมูล
        epochs=3,                       # ตั้งไว้ 3 รอบเพื่อให้เสร็จไวบน CPU (หรือเปลี่ยนเป็น 1 ตามต้องการ)
        batch=16,                       # ขนาด Batch Size เหมาะสมกับแรม 16GB
        imgsz=640,                      # ขนาดรูปภาพ
        name="fruit_model_3classes"      # โฟลเดอร์เก็บผลลัพธ์
    )

    print("การเทรนเสร็จสมบูรณ์!")

if __name__ == '__main__':
    main()
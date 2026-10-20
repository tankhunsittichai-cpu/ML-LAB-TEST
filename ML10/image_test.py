from ultralytics import YOLO
import glob

def main():
    # โหลดโมเดลที่เทรนเสร็จแล้ว
    model = YOLO("runs/detect/fruit_model_3classes/weights/best.pt")

    # ค้นหารูปภาพทั้งหมดในโฟลเดอร์ test/images อัตโนมัติ
    image_files = glob.glob("data/test/images/*.jpg") + glob.glob("data/test/images/*.png")

    if not image_files:
        print("ไม่พบรูปภาพในโฟลเดอร์ data/test/images/ กรุณาตรวจสอบอีกครั้ง")
        return

    # เลือกรูปแรกที่เจอมาทดสอบ
    image_path = image_files[0]
    print(f"กำลังทดสอบรูปภาพ: {image_path}")

    # ทำนายผลลัพธ์
    results = model(image_path, show=True, save=True)

    print("ทดสอบรูปภาพเสร็จสิ้น! ตรวจสอบผลลัพธ์ได้ในโฟลเดอร์ runs/predict")

if __name__ == '__main__':
    main()
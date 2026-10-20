from ultralytics import YOLO
import cv2

def main():
    # โหลดโมเดลที่เทรนเสร็จแล้ว
    model = YOLO("runs/detect/fruit_model_3classes/weights/best.pt")

    # เปิดกล้องเว็บแคม (0 คือกล้องหลัก)
    cap = cv2.VideoCapture(0)

    print("กำลังเปิดกล้อง... กดปุ่ม 'q' เพื่อปิดหน้าต่าง")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจจับวัตถุแบบ Real-time
        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("YOLOv8 Real-time Fruit Detection", annotated_frame)

        # กด 'q' เพื่อหยุดการทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
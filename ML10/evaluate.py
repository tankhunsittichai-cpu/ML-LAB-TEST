from ultralytics import YOLO

def main():
    # โหลดโมเดลที่เทรนเสร็จแล้ว
    model = YOLO("runs/detect/fruit_model_3classes/weights/best.pt")

    # ประเมินผลโมเดลกับชุดข้อมูล
    metrics = model.val(data="data.yaml")

    print("--- ผลการประเมินโมเดล ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")

if __name__ == '__main__':
    main()
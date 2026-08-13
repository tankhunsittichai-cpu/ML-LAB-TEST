import os
 
def create_output_folder(folder_name="output"):
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, folder_name)
 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"สร้างโฟลเดอร์ {folder_name} เรียบร้อยแล้ว")
    else:
        print(f"โฟลเดอร์ {folder_name} มีอยู่แล้ว")
 
    return output_path
 
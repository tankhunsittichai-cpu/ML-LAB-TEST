import pandas as pd

def load_data(filepath):
    """ฟังก์ชันสำหรับโหลดชุดข้อมูล"""
    df = pd.read_csv(filepath)
    return df
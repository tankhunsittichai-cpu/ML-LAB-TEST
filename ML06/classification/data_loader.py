import pandas as pd

def load_data(filepath):
    """โหลดข้อมูลจากไฟล์ CSV"""
    df = pd.read_csv(filepath)
    return df
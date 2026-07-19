import pandas as pd

# 1. โหลดข้อมูลเข้ามาก่อนเพื่อให้ระบบรู้จักตัวแปร df
df = pd.read_csv(r"C:\Users\Admin\Desktop\ML-tase\ML02\sample_-_superstore.csv", encoding="cp874")

print("="*60)
print("PART 4: FEATURE ENGINEERING (USING PANDAS)")
print("="*60)

# 2. ทำ Label Encoding ด้วย Pandas
print("1. [Process] กำลังทำ Label Encoding สำหรับคอลัมน์ 'Segment'...")
df['Segment_Encoded'] = df['Segment'].astype('category').cat.codes

print("\n--- ผลลัพธ์หลังทำ Label Encoding ---")
print(df[['Segment', 'Segment_Encoded']].drop_duplicates().reset_index(drop=True))


# 3. ทำ One-Hot Encoding ด้วย Pandas
print("\n2. [Process] กำลังทำ One-Hot Encoding สำหรับคอลัมน์ 'Ship Mode'...")
df_one_hot = pd.get_dummies(df['Ship Mode'], prefix='ShipMode', dtype=int)
df = pd.concat([df, df_one_hot], axis=1)

print("\n--- ผลลัพธ์หลังทำ One-Hot Encoding (คอลัมน์ที่เพิ่มขึ้นมา) ---")
print(df_one_hot.head())

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE!")
print("="*60)
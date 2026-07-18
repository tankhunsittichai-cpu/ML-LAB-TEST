import pandas as pd

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\Admin\Desktop\ML-tase\sample_-_superstore.csv", encoding="cp874")


print("1. LOAD DATASET")
print(df.head())

# 2. Display Shape (เพิ่มตรงนี้ครับ)
print("="*50)
print("2. DISPLAY DATASET SHAPE")
print("จำนวน (แถว, คอลัมน์) คือ:", df.shape)

# 3. Display Data Types (เพิ่มตรงนี้ครับ)
print("="*50)
print("3. DISPLAY DATA TYPES")
print(df.dtypes)  # หรือจะเปลี่ยนเป็น df.info() ก็ได้ครับ

# 4. Display Summary Statistics (เพิ่มตรงนี้ครับ)
print("="*50)
print("4. DISPLAY SUMMARY STATISTICS")
print(df.describe())

# 5. Display Missing Values (เพิ่มตรงนี้ครับ)
print("="*50)
print("5. DISPLAY MISSING VALUES")
print(df.isnull().sum())

# 6. Display Duplicate Records (เพิ่มตรงนี้ครับ)
print("="*50)
print("6. DISPLAY DUPLICATE RECORDS")
print("จำนวนแถวที่ข้อมูลซ้ำกันมีทั้งหมด:", df.duplicated().sum(), "แถว")

# 7. Display Class Distribution (เพิ่มตรงนี้ครับ)
print("="*50)
print("7. DISPLAY CLASS DISTRIBUTION")
# เปลี่ยนคำว่า 'Category' เป็นชื่อคอลัมน์อื่นๆ ในไฟล์ของคุณได้ตามต้องการครับ
print(df['Category'].value_counts())
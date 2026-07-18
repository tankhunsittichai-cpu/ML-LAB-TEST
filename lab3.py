import pandas as pd

# --- โหลดข้อมูลเริ่มต้น ---
df = pd.read_csv(r"C:\Users\Admin\Desktop\ML-tase\sample_-_superstore.csv", encoding="cp874")

print("="*60)
print("PART 3: DATA CLEANING & STATS COMPARISON")
print(f"ขนาดข้อมูลเริ่มต้น: {df.shape}")
print("="*60)

# ==========================================================
# [ก่อนทำความสะอาด] คำนวณ Mean และ Median เก็บไว้ก่อน
# ==========================================================
# เลือกคอลัมน์ที่เป็นตัวเลขหลักมาเปรียบเทียบ เช่น Sales และ Profit
mean_before = df[['Sales', 'Profit']].mean()
median_before = df[['Sales', 'Profit']].median()


# ==========================================================
# PERFORM DATA CLEANING
# ==========================================================

# 1. Missing Value Handling
print("1. [Process] กำลังจัดการ Missing Values...")
if 'Postal Code' in df.columns:
    df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()[0])
df.dropna(inplace=True) # ลบแถวที่มีค่าว่างที่เหลือทิ้ง


# 2. Duplicate Removal
print("2. [Process] กำลังลบข้อมูลที่ซ้ำซ้อน...")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)


# 3. Incorrect Data Correction
print("3. [Process] กำลังแก้ไขข้อความสะกดผิดและลบช่องว่างส่วนเกิน...")
# ลบช่องว่างหน้า-หลังข้อความในทุกคอลัมน์ตัวหนังสือ
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# ตัวอย่างการแก้คำสะกดผิด (หากมี)
if 'Segment' in df.columns:
    df['Segment'] = df['Segment'].replace({'Home Ofice': 'Home Office'})


# 4. Data Type Conversion (เพิ่มใหม่)
print("4. [Process] กำลังแปลงประเภทข้อมูล (Data Type Conversion)...")
# ตัวอย่างที่ 1: แปลงวันที่จาก Object เป็น Datetime (เพื่อให้ใช้วิเคราะห์เวลาได้ถูกต้อง)
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# ตัวอย่างที่ 2: แปลงคอลัมน์รหัสให้เป็นหมวดหมู่ (Category) หรือตัวหนังสือ (String) เพื่อประหยัดเมมโมรี่
if 'Postal Code' in df.columns:
    df['Postal Code'] = df['Postal Code'].astype(str)


# ==========================================================
# [หลังทำความสะอาด] คำนวณ Mean และ Median ใหม่
# ==========================================================
mean_after = df[['Sales', 'Profit']].mean()
median_after = df[['Sales', 'Profit']].median()


# ==========================================================
# COMPARE MEAN & MEDIAN (แสดงการเปรียบเทียบ)
# ==========================================================
print("\n" + "="*20 + " COMPARE STATS " + "="*20)

# ทำเป็นตารางเปรียบเทียบค่า Mean
print("\n--- Comparison of MEAN (ค่าเฉลี่ย) ---")
compare_mean = pd.DataFrame({'Before Clean': mean_before, 'After Clean': mean_after})
compare_mean['Difference'] = compare_mean['After Clean'] - compare_mean['Before Clean']
print(compare_mean)

# ทำเป็นตารางเปรียบเทียบค่า Median
print("\n--- Comparison of MEDIAN (ค่ามัธยฐาน) ---")
compare_median = pd.DataFrame({'Before Clean': median_before, 'After Clean': median_after})
compare_median['Difference'] = compare_median['After Clean'] - compare_median['Before Clean']
print(compare_median)

print("\n" + "="*55)
print(f"ขนาดข้อมูลสุดท้ายหลังทำความสะอาด: {df.shape}")
print("="*55)
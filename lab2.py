import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- โหลดข้อมูลจาก LAB1 ---
df = pd.read_csv(r"C:\Users\Admin\Desktop\ML-tase\sample_-_superstore.csv", encoding="cp874")

print("="*50)
print("LAB 2: DATA VISUALIZATION STARTING...")
print("="*50)

# --- 1. Histogram ---

df['Sales'].hist(bins=30, color='skyblue', rwidth=1)
plt.title('Histogram of Sales Distribution', fontsize=14)
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show() # กราฟแรกจะเด้งขึ้นมา พอคุณกดปิดหน้าต่างกราฟนี้...

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.show() # ...กราฟที่สองถึงจะเด้งขึ้นมาแสดงต่อครับ
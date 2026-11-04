import pandas as pd

# ==========================================
# 1. IDENTIFIKASI DATASET IRIS
# ==========================================

df_iris = pd.read_csv("Dataset/Multiclass/IRIS.csv")

print("\n========== DATASET IRIS ==========")
print("Jumlah Data:", df_iris.shape[0])
print("Jumlah Kolom:", df_iris.shape[1])
print("Nama Kolom:", df_iris.columns.tolist())
print("\nMissing Values:")
print(df_iris.isnull().sum())
print("\nDistribusi Kelas:")
print(df_iris["species"].value_counts())

print("""
KESIMPULAN:
Dataset Iris termasuk Multiclass Classification.
Memiliki 150 data, 5 kolom, dan 3 kelas spesies bunga.
Dataset tidak memiliki missing values.
""")


# ==========================================
# 2. IDENTIFIKASI DATASET CAR EVALUATION
# ==========================================

df_car = pd.read_csv(
    "Dataset/Multiclass/car_evaluation.csv",
    header=None,
    names=[
        "buying", "maint", "doors",
        "persons", "lug_boot", "safety", "class"
    ]
)

print("\n========== DATASET CAR EVALUATION ==========")
print("Jumlah Data:", df_car.shape[0])
print("Jumlah Kolom:", df_car.shape[1])
print("Nama Kolom:", df_car.columns.tolist())
print("\nMissing Values:")
print(df_car.isnull().sum())
print("\nDistribusi Kelas:")
print(df_car["class"].value_counts())

print("""
KESIMPULAN:
Dataset Car Evaluation termasuk Multiclass Classification.
Memiliki 1.728 data, 7 kolom, dan 4 kelas evaluasi mobil.
Dataset tidak memiliki missing values.
""")

# ==========================================
# 3. IDENTIFIKASI DATASET GLASS
# ==========================================

df_glass = pd.read_csv(
    "Dataset/Multiclass/archive (1)/glass.csv"
)

print("\n========== DATASET GLASS ==========")
print("Jumlah Data:", df_glass.shape[0])
print("Jumlah Kolom:", df_glass.shape[1])
print("Nama Kolom:", df_glass.columns.tolist())
print("\nMissing Values:")
print(df_glass.isnull().sum())

print("\nDistribusi Kelas:")
print(df_glass.iloc[:, -1].value_counts().sort_index())

print("""
KESIMPULAN:
Dataset Glass Identification termasuk Multiclass Classification.
Memiliki 214 data, 10 kolom, dan 6 kelas jenis kaca.
Dataset tidak memiliki missing values.
""")
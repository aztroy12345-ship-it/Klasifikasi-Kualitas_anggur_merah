# ============================================================
# IDENTIFIKASI DATASET HEART DISEASE
# Jenis: Binary Classification
# Sumber: Kaggle
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1. MEMBACA DATASET
# ------------------------------------------------------------

file_path = "Dataset/Binary/archive/heart_disease_health_indicators_BRFSS2015.csv"

df = pd.read_csv(file_path)

# ------------------------------------------------------------
# 2. IDENTIFIKASI DATASET
# ------------------------------------------------------------

print("=== IDENTIFIKASI DATASET HEART DISEASE ===")

print("Jumlah Data:", df.shape[0])
print("Jumlah Kolom:", df.shape[1])

# ------------------------------------------------------------
# 3. MENAMPILKAN DATA PERTAMA
# ------------------------------------------------------------

print("\n=== 5 DATA PERTAMA ===")
print(df.head())

# ------------------------------------------------------------
# 4. INFORMASI DATASET
# ------------------------------------------------------------

print("\n=== INFORMASI DATASET ===")
df.info()

# ------------------------------------------------------------
# 5. MEMERIKSA MISSING VALUES
# ------------------------------------------------------------

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ------------------------------------------------------------
# 6. IDENTIFIKASI TARGET
# ------------------------------------------------------------

target = "HeartDiseaseorAttack"

print("\n=== DISTRIBUSI TARGET ===")
print(df[target].value_counts())

# ------------------------------------------------------------
# 7. REFERENSI DATASET
# ------------------------------------------------------------

print("\nSumber Dataset:")
print("Kaggle - Heart Disease Health Indicators Dataset")
print("https://www.kaggle.com/datasets/alexteboul/")
print("heart-disease-health-indicators-dataset")

# ============================================================
# 8. KESIMPULAN IDENTIFIKASI
# ============================================================


# ============================================================
# IDENTIFIKASI DATASET TITANIC
# Jenis: Binary Classification
# Sumber: Kaggle
# ============================================================

import pandas as pd

# Membaca dataset Titanic
df = pd.read_csv("Dataset/Binary/titanic/train.csv")

# Identifikasi dataset
print("=== IDENTIFIKASI DATASET TITANIC ===")
print("Jumlah Data:", df.shape[0])
print("Jumlah Kolom:", df.shape[1])

# Menampilkan 5 data pertama
print("\n=== 5 DATA PERTAMA ===")
print(df.head())

# Informasi dataset
print("\n=== INFORMASI DATASET ===")
df.info()

# Memeriksa missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Distribusi target
print("\n=== DISTRIBUSI TARGET ===")
print(df["Survived"].value_counts())

# Referensi
print("\nSumber Dataset:")
print("Kaggle - Titanic")
print("https://www.kaggle.com/competitions/titanic")

# Kesimpulan
print("\n=== KESIMPULAN ===")
print("""
Dataset Titanic memiliki 891 data dan 12 kolom.
Target Survived memiliki dua kelas:
0 = Tidak selamat
1 = Selamat

Dataset termasuk Binary Classification.
Dataset memiliki missing values pada beberapa atribut.
""")


# ============================================================
# IDENTIFIKASI DATASET BANKNOTE AUTHENTICATION
# Jenis: Binary Classification
# Sumber: Kaggle
# ============================================================

import pandas as pd

# Membaca dataset
df = pd.read_csv("Dataset/Binary/BankNote_Authentication.csv")

# Identifikasi dataset
print("=== IDENTIFIKASI DATASET BANKNOTE ===")
print("Jumlah Data:", df.shape[0])
print("Jumlah Kolom:", df.shape[1])

# Menampilkan 5 data pertama
print("\n=== 5 DATA PERTAMA ===")
print(df.head())

# Informasi dataset
print("\n=== INFORMASI DATASET ===")
df.info()

# Memeriksa missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Distribusi target
print("\n=== DISTRIBUSI TARGET ===")
print(df["class"].value_counts())

# Referensi
print("\nSumber Dataset:")
print("Kaggle - Bank Note Authentication")
print("https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data")

# Kesimpulan
print("\n=== KESIMPULAN ===")
print("""
Dataset BankNote Authentication memiliki 1.372 data
dan 5 kolom, terdiri dari 4 atribut fitur dan
1 atribut target yaitu class.

Target class memiliki dua label:
0 = Uang asli
1 = Uang palsu

Dataset termasuk Binary Classification
dan tidak memiliki missing values.
""")


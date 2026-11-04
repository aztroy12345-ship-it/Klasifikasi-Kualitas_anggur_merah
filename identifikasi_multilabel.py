import pandas as pd
from scipy.io import arff

# ==========================================
# 1. IDENTIFIKASI MOVIE GENRE
# ==========================================
df_movie = pd.read_csv(
    "Dataset/Multilabel/archive (3)/MovieGenre.csv",
    encoding="latin1"
)

print("\n========== MOVIE GENRE ==========")
print("Jumlah Data:", df_movie.shape[0])
print("Jumlah Kolom:", df_movie.shape[1])
print("Nama Kolom:", df_movie.columns.tolist())
print("\nMissing Values:")
print(df_movie.isnull().sum())

# Genre dipisahkan menggunakan tanda |
genre = df_movie["Genre"].dropna().str.split("|")
print("\nJumlah Genre Unik:", len(set(g for row in genre for g in row)))
print("\nContoh Genre:")
print(genre.head())

print("""
KESIMPULAN:
Movie Genre termasuk Multilabel Classification
karena satu film dapat memiliki beberapa genre sekaligus.
""")

# ==========================================
# 2. IDENTIFIKASI DATASET ARXIV
# ==========================================

df_arxiv = pd.read_csv(
    "Dataset/Multilabel/archive (4)/train.csv"
)

print("\n========== ARXIV PAPERS ==========")
print("Jumlah Data:", df_arxiv.shape[0])
print("Jumlah Kolom:", df_arxiv.shape[1])
print("Nama Kolom:", df_arxiv.columns.tolist())
print("\nMissing Values:")
print(df_arxiv.isnull().sum())

print("\nContoh Data:")
print(df_arxiv.head())

print("""
KESIMPULAN:
Dataset ArXiv termasuk Multilabel Classification
karena satu artikel dapat memiliki beberapa kategori penelitian.
""")

from scipy.io import arff
import pandas as pd

data, metadata = arff.loadarff(
    "Dataset/Multilabel/emotions-test.arff"
)

df_emotions = pd.DataFrame(data)

print("\n========== DATASET EMOTIONS ==========")
print("Jumlah Data:", df_emotions.shape[0])
print("Jumlah Kolom:", df_emotions.shape[1])
print("Nama Kolom:", df_emotions.columns.tolist())

print("\nMissing Values:")
print(df_emotions.isnull().sum())

print("\nDistribusi Label:")
for kolom in df_emotions.columns[-6:]:
    print(kolom, df_emotions[kolom].value_counts().to_dict())

print("""
KESIMPULAN:
Dataset Emotions termasuk Multilabel Classification
karena satu data audio dapat memiliki beberapa label emosi.
""")

print("""
REFERENSI:
https://www.openml.org/search?id=41465&type=data
""")

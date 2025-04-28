import pandas as pd
import matplotlib.pyplot as plt

# Membaca file CSV
df = pd.read_csv('data.csv')

# Membuat plot histogram kolom "Duration"
plt.figure(figsize=(8, 6))  # opsional, biar ukuran grafik lebih enak dilihat
df["Duration"].plot(kind='hist', bins=20, edgecolor='black')

# Menambahkan judul dan label
plt.title('Histogram of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')

# Menampilkan plot
plt.show()

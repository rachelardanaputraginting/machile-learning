import pandas as pd

def tampilkan_menu():
    print("\n=== Menu Statistik BloodPressure ===")
    print("1. Tampilkan Minimum")
    print("2. Tampilkan Maximum")
    print("3. Tampilkan Rata-rata")
    print("4. Tampilkan Median")
    print("5. Tampilkan Semua Statistik")
    print("6. Keluar")

def hitung_statistik(df):
    sum_bp = 0
    min_bp = df.iloc[0]['BloodPressure']
    max_bp = df.iloc[0]['BloodPressure']
    count = 0
    values = []

    for i in range(len(df)):
        bp = df.iloc[i]['BloodPressure']
        sum_bp += bp
        count += 1
        values.append(bp)
        if bp < min_bp:
            min_bp = bp
            print(f"Minimum", {i})
        if bp > max_bp:
            max_bp = bp
            print(i)

    mean_bp = sum_bp / count
    values.sort()
    n = len(values)
    if n % 2 == 0:
        median_bp = (values[n//2 - 1] + values[n//2]) / 2
    else:
        median_bp = values[n//2]

    return {
        'min': min_bp,
        'max': max_bp,
        'mean': mean_bp,
        'median': median_bp
    }

# Membaca file CSV
df = pd.read_csv('dataset/diabetes.csv')

# 1. Menampilkan 2 baris menggunakan iloc (berdasarkan posisi)
print("▶️ Baris ke-2 dan ke-4 (berdasarkan posisi):")
print(df.iloc[[1, 3]])
print("\n")

# 2. Menampilkan baris dengan index antara 2 dan 9 menggunakan .isin()
print("▶️ Baris dengan index antara 2 dan 9 (dengan .isin()):")
print(df[df.index.isin(range(2, 10))])
print("\n")

# 3. Mengubah index agar dimulai dari 1 agar mudah dengan loc
df.index = range(1, len(df) + 1)

# 4. Menggunakan .loc[] untuk ambil 1 baris berdasarkan label
print("▶️ df.loc[1] (berdasarkan label):")
print(df.loc[1])
print("\n")

# 5. Menggunakan .iloc[] untuk ambil 1 baris berdasarkan posisi
print("▶️ df.iloc[1] (berdasarkan posisi):")
print(df.iloc[1])
print("\n")

# 6. Menggunakan .loc[] untuk ambil range baris (label)
print("▶️ df.loc[100:110] (label 100 sampai 110):")
print(df.loc[100:110])
print("\n")

# 7. Menggunakan .iloc[] untuk ambil range baris (posisi)
print("▶️ df.iloc[100:110] (posisi 100 sampai 109):")
print(df.iloc[100:110])
print("\n")

# 8. Menggunakan .loc[] dengan list index
print("▶️ df.loc[[100, 200, 300]]:")
print(df.loc[[100, 200, 300]])
print("\n")

# 9. Menggunakan .iloc[] dengan list posisi
print("▶️ df.iloc[[100, 200, 300]]:")
print(df.iloc[[100, 200, 300]])
print("\n")

# 10. Menggunakan .loc[] untuk ambil baris dan kolom spesifik
print("▶️ df.loc[100:110, ['Pregnancies', 'Glucose', 'BloodPressure']]:")
print(df.loc[100:110, ['Pregnancies', 'Glucose', 'BloodPressure']])
print("\n")

# 11. Menggunakan .iloc[] untuk ambil baris dan kolom spesifik (posisi)
print("▶️ df.iloc[100:110, :3]:")
print(df.iloc[100:110, :3])
print("\n")

# 12. Mengambil baris dari 760 sampai akhir dan kolom tertentu
print("▶️ df.loc[760:, ['Pregnancies', 'Glucose', 'BloodPressure']]:")
print(df.loc[760:, ['Pregnancies', 'Glucose', 'BloodPressure']])
print("\n")

print("▶️ df.iloc[760:, :3]:")
print(df.iloc[760:, :3])
print("\n")

# 13. Mengubah nilai pada baris tertentu
print("▶️ Mengubah nilai kolom 'Age' dari 81 menjadi 80 (jika ada)...")
df.loc[df['Age'] == 81, ['Age']] = 80
print("Perubahan selesai.\n")

# 14. Tampilkan hasil baris dengan Age 80 sebagai verifikasi
print("▶️ Baris dengan Age == 80:")
print(df[df['Age'] == 80])
print("\n")

# 15. Isolasi baris berdasarkan kondisi sederhana
print("▶️ Baris dengan BloodPressure == 122:")
print(df[df['BloodPressure'] == 122])
print("\n")

print("▶️ Baris dengan Outcome == 1:")
print(df[df['Outcome'] == 1])
print("\n")

print("▶️ Baris dengan BloodPressure > 100 (hanya kolom tertentu):")
print(df.loc[df['BloodPressure'] > 100, ['Pregnancies', 'Glucose', 'BloodPressure']])
print("\n")

# 16. Menampilkan semua baris menggunakan iloc dengan perulangan for
print("▶️ Menampilkan semua baris satu per satu menggunakan iloc dan for loop:\n")
for i in range(len(df)):
    print(f"Baris ke-{i}:")
    print(df.iloc[i])
    print("-" * 50)

# 17. Menampilkan statistik BloodPressure dengan menu
print("\n▶️ Statistik BloodPressure:")
stats = hitung_statistik(df.head(10))


while True:
    tampilkan_menu()
    pilihan = input("\nPilih operasi (1-6): ")
    
    if pilihan == '1':
        print(f"Minimum BloodPressure: {stats['min']}")
    elif pilihan == '2':
        print(f"Maximum BloodPressure: {stats['max']}")
    elif pilihan == '3':
        print(f"Rata-rata BloodPressure: {stats['mean']:.2f}")
    elif pilihan == '4':
        print(f"Median BloodPressure: {stats['median']}")
    elif pilihan == '5':
        print(f"Minimum BloodPressure: {stats['min']}")
        print(f"Maximum BloodPressure: {stats['max']}")
        print(f"Rata-rata BloodPressure: {stats['mean']:.2f}")
        print(f"Median BloodPressure: {stats['median']}")
    elif pilihan == '6':
        print("Terima kasih!")
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")

# 18. Menampilkan nilai Rata-rata dari kolom BloodPressure 

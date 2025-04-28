import pandas as pd

def tampilkan_menu():
    print("\n=== ALAT PEMBERSIHAN DATA ===")
    print("1. Hapus Sel Kosong")
    print("2. Perbaiki Format yang Salah (Tanggal)")
    print("3. Perbaiki Data yang Salah (Durasi > 120)")
    print("4. Hapus Duplikat")
    print("5. Tampilkan Data Saat Ini")
    print("6. Keluar")

def remove_empty_cells(df):
    print("\nMenghapus sel kosong...")
    new_df = df.dropna()
    print("Sel kosong berhasil dihapus!")
    return new_df

def fix_date_format(df):
    print("\nMemperbaiki format tanggal...")
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        print("Format tanggal berhasil diperbaiki!")
    except Exception as e:
        print(f"Error saat memperbaiki format tanggal: {e}")
    return df

def fix_wrong_duration(df):
    print("\nMemperbaiki data durasi yang salah...")
    # Pertama, batasi nilai maksimal 120
    for x in df.index:
        if df.loc[x, "Duration"] > 120:
            df.loc[x, "Duration"] = 120
    
    # Kemudian hapus baris yang durasinya masih > 120
    for x in df.index:
        if df.loc[x, "Duration"] > 120:
            df.drop(x, inplace=True)
    
    print("Data durasi berhasil diperbaiki!")
    return df

def remove_duplicates(df):
    print("\nMenghapus duplikat...")
    print("Baris duplikat ditemukan:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print("Duplikat berhasil dihapus!")
    return df

def show_data(df):
    print("\nData Saat Ini:")
    print(df.to_string())

def main():
    try:
        df = pd.read_csv('../dataset/data.csv')
        print("Data berhasil dimuat!")
        
        while True:
            tampilkan_menu()
            pilihan = input("\nPilih operasi (1-6): ")
            
            if pilihan == '6':
                print("Terima kasih telah menggunakan Alat Pembersihan Data!")
                break
                
            elif pilihan == '1':
                df = remove_empty_cells(df)
            elif pilihan == '2':
                df = fix_date_format(df)
            elif pilihan == '3':
                df = fix_wrong_duration(df)
            elif pilihan == '4':
                df = remove_duplicates(df)
            elif pilihan == '5':
                show_data(df)
            else:
                print("Pilihan tidak valid. Silakan coba lagi.")
            
            # Simpan perubahan setelah setiap operasi
            df.to_csv('../dataset/cleaned_data.csv', index=False)
            
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main() 
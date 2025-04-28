import pandas as pd
import matplotlib.pyplot as plt

def tampilkan_menu():
    print("\n=== ALAT ANALISIS DATA ===")
    print("1. Tampilkan Korelasi")
    print("2. Tampilkan Plot")
    print("3. Tampilkan Scatter Plot")
    print("4. Tampilkan Histogram")
    print("5. Keluar")

def show_correlation(df):
    print("\nMenampilkan korelasi antar kolom:")
    correlation = df.corr()
    print(correlation)
    return df

def show_plot(df):
    print("\nMenampilkan plot data:")
    df.plot()
    plt.title('Data Plot')
    plt.tight_layout()
    plt.show()
    return df

def show_scatter_plot(df):
    print("\nMenampilkan scatter plot:")
    df.plot(kind='scatter', x='Duration', y='Calories')
    plt.title('Scatter Plot: Duration vs Calories')
    plt.tight_layout()
    plt.show()
    return df

def show_histogram(df):
    print("\nMenampilkan histogram:")
    plt.figure(figsize=(8, 6))
    df["Duration"].plot(kind='hist', bins=20, edgecolor='black')
    plt.title('Histogram of Duration')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return df

def main():
    try:
        # Langsung menggunakan file data.csv
        df = pd.read_csv('data.csv')
        print("Data berhasil dimuat!")
        
        while True:
            tampilkan_menu()
            pilihan = input("\nPilih operasi (1-5): ")
            
            if pilihan == '5':
                print("Terima kasih telah menggunakan Alat Analisis Data!")
                break
                
            elif pilihan == '1':
                df = show_correlation(df)
            elif pilihan == '2':
                df = show_plot(df)
            elif pilihan == '3':
                df = show_scatter_plot(df)
            elif pilihan == '4':
                df = show_histogram(df)
            else:
                print("Pilihan tidak valid. Silakan coba lagi.")
            
    except FileNotFoundError:
        print("File 'data.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan program.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
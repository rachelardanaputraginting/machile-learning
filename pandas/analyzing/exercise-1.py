import pandas as pd

def tampilkan_menu():
    print("=== ANALYSIS DATA ===")
    print("1. Head") # Operasi penjumlahan
    print("2. Tail") # Operasi pengurangan
    print("3. Info") # Operasi perkalian
    print("4. Keluar") # Pilihan untuk keluar dari program

def head(total): 
    return total


def tail(total): 
    return total

while True:
    tampilkan_menu()

    df = pd.read_csv("../dataset/diabetes.csv")

    pilihan = input("Pilih operasi (1/2/3/4): ")
    if pilihan == '4':
        print("Terimakasih sudah analisis data")
        break
    
    if pilihan not in ['1', '2', '3', '4']:
        print("Pilihan tidak valid. Silakan coba lagi.")
        continue  # Kembali ke awal loop

    if pilihan == '1':  # Head
        total = int(input("Masukkan total: "))
        print(df.head(total))
    elif pilihan == '2':  # Tail
        total = int(input("Masukkan total: "))
        print(df.tail(total))
    elif pilihan == '3':  # Info
        print(df.info())

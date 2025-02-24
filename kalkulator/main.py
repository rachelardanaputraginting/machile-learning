# Program Kalkulator Sederhana

# Fungsi untuk menampilkan menu pilihan operasid
def tampilkan_menu():
    print("=== KALKULATOR SEDERHANA ===")
    print("1. Penjumlahan (+)") # Operasi penjumlahan
    print("2. Pengurangan (-)") # Operasi pengurangan
    print("3. Perkalian (*)") # Operasi perkalian
    print("4. Pembagian (/)") # Operasi pembagian
    print("5. Modulus (%)") # Operasi modulus
    print("6. Keluar") # Operasi modulus

# Fungsi untuk melakukan penjumlahan
def tambah(a, b):
    return a + b # Mengambalikan hasil penjumlahan a + b

def kurang(a, b):
    return a - b # Mengambalikan hasil penjumlahan a + b

def kali(a, b):
    return a * b # Mengambalikan hasil penjumlahan a + b

def bagi(a, b):
    if b == 0:
        return "Error: Pembagian dengan nol tidak diperbolehkan!"
    return a / b # Mengambalikan hasil penjumlahan a + b

def modulus(a, b):
    if b == 0:
        return "Error: Modulus dengan nol tidak diperbolehkan!"
    return a % b # Mengambalikan hasil penjumlahan a + b

# Program utama
while True:
    tampilkan_menu()

    pilihan = input("Pilih operasi (1/2/3/4/5/6): ")

    if pilihan == '6':
        print("Terimakasih telah menggunakan kalkulator ini!")
        break  # Keluar dari loop
    
    if pilihan not in ['1', '2', '3', '4', '5']:
        print("Pilihan tidak valid. Silakan coba lagi.")
        continue  # Kembali ke awal loop

    try:
        angka1 = float(input("Masukkan angka pertama: "))
        angka2 = float(input("Masukkan angka kedua: "))
    except:
        print("Input harus berupa angka. Silahkan coba lagi.")
        continue

    if pilihan == '1':
        hasil = tambah(angka1, angka2)
        print(f"Hasil: {angka1} + {angka2} = {hasil}")
    elif pilihan == '2':
        hasil = kurang(angka1, angka2)
        print(f"Hasil: {angka1} - {angka2} = {hasil}")
    elif pilihan == '3':
        hasil = kali(angka1, angka2)
        print(f"Hasil: {angka1} * {angka2} = {hasil}")
    elif pilihan == '4':
        hasil = bagi(angka1, angka2)
        print(f"Hasil: {angka1} / {angka2} = {hasil}")
    elif pilihan == '5':
        hasil = modulus(angka1, angka2)
        print(f"Hasil: {angka1} % {angka2} = {hasil}")
    
    print() 


# Program Kalkulator Sederhana

# Fungsi untuk menampilkan menu pilihan operasi
def tampilkan_menu():
    print("=== KALKULATOR SEDERHANA ===")
    print("1. Penjumlahan (+)")  # Operasi penjumlahan
    print("2. Pengurangan (-)")  # Operasi pengurangan
    print("3. Perkalian (*)")     # Operasi perkalian
    print("4. Pembagian (/)")     # Operasi pembagian
    print("5. Modulus (%)")       # Operasi modulus (sisa bagi)
    print("6. Pangkat (^)")       # Operasi pangkat (eksponen) menggunakan perulangan
    print("7. Akar Kuadrat (√)")  # Operasi akar kuadrat menggunakan perulangan
    print("8. Keluar")            # Pilihan untuk keluar dari program

# Fungsi untuk melakukan penjumlahan
def tambah(a, b):
    return a + b  # Mengembalikan hasil penjumlahan a dan b

# Fungsi untuk melakukan pengurangan
def kurang(a, b):
    return a - b  # Mengembalikan hasil pengurangan a dan b

# Fungsi untuk melakukan perkalian
def kali(a, b):
    return a * b  # Mengembalikan hasil perkalian a dan b

# Fungsi untuk melakukan pembagian
def bagi(a, b):
    if b == 0:  # Memeriksa apakah pembagi adalah 0
        return "Error: Pembagian dengan nol tidak diperbolehkan!"
    return a / b  # Mengembalikan hasil pembagian a dan b

# Fungsi untuk melakukan modulus
def modulus(a, b):
    if b == 0:  # Memeriksa apakah pembagi adalah 0
        return "Error: Modulus dengan nol tidak diperbolehkan!"
    return a % b  # Mengembalikan hasil modulus a dan b

# Fungsi untuk melakukan pangkat menggunakan perulangan
def pangkat(a, b):
    if b == 0:  # Jika pangkatnya 0, hasilnya selalu 1 (a^0 = 1)
        return 1
    elif b > 0:  # Jika pangkat positif
        hasil = 1
        for _ in range(b):  # Melakukan perulangan sebanyak nilai pangkat (b)
            hasil *= a  # Mengalikan hasil dengan a setiap iterasi
        return hasil
    else:  # Jika pangkat negatif
        hasil = 1
        for _ in range(-b):  # Melakukan perulangan sebanyak nilai pangkat (b), tapi positif
            hasil *= a  # Mengalikan hasil dengan a setiap iterasi
        return 1 / hasil  # Mengembalikan hasil sebagai pecahan (1 / hasil)

# Fungsi untuk menghitung akar kuadrat menggunakan Metode Newton-Raphson
def akar_kuadrat(a, toleransi=1e-10, max_iter=1000):
    if a < 0:  # Memeriksa apakah input valid (tidak boleh negatif)
        return "Error: Tidak dapat menghitung akar kuadrat dari bilangan negatif."
    
    tebakan = a  # Tebakan awal adalah nilai a itu sendiri
    for _ in range(max_iter):  # Melakukan iterasi maksimal max_iter kali
        tebakan_baru = 0.5 * (tebakan + a / tebakan)  # Rumus Newton-Raphson
        if abs(tebakan_baru - tebakan) < toleransi:  # Cek apakah sudah cukup dekat
            return tebakan_baru
        tebakan = tebakan_baru  # Update tebakan
    return tebakan  # Mengembalikan hasil akhir jika iterasi maksimal tercapai

# Program utama
while True:  # Looping agar program berjalan terus sampai user memilih keluar
    tampilkan_menu()  # Menampilkan menu pilihan operasi
    
    # Meminta input pilihan operasi dari user
    pilihan = input("Pilih operasi (1/2/3/4/5/6/7/8): ")
    
    # Jika user memilih keluar
    if pilihan == '8':
        print("Terima kasih telah menggunakan kalkulator ini!")
        break  # Keluar dari loop
    
    # Memastikan pilihan valid
    if pilihan not in ['1', '2', '3', '4', '5', '6', '7']:
        print("Pilihan tidak valid. Silakan coba lagi.")
        continue  # Kembali ke awal loop
    
    # Meminta input angka pertama
    try:
        angka1 = float(input("Masukkan angka: "))  # Input angka pertama
    except ValueError:  # Menangani kesalahan jika input bukan angka
        print("Input harus berupa angka. Silakan coba lagi.")
        continue  # Kembali ke awal loop
    
    # Melakukan operasi sesuai pilihan user
    if pilihan == '1':  # Penjumlahan
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = tambah(angka1, angka2)
        print(f"Hasil: {angka1} + {angka2} = {hasil}")
    elif pilihan == '2':  # Pengurangan
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = kurang(angka1, angka2)
        print(f"Hasil: {angka1} - {angka2} = {hasil}")
    elif pilihan == '3':  # Perkalian
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = kali(angka1, angka2)
        print(f"Hasil: {angka1} * {angka2} = {hasil}")
    elif pilihan == '4':  # Pembagian
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = bagi(angka1, angka2)
        print(f"Hasil: {angka1} / {angka2} = {hasil}")
    elif pilihan == '5':  # Modulus
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = modulus(angka1, angka2)
        print(f"Hasil: {angka1} % {angka2} = {hasil}")
    elif pilihan == '6':  # Pangkat
        angka2 = int(input("Masukkan pangkat (bilangan bulat): "))
        hasil = pangkat(angka1, angka2)
        print(f"Hasil: {angka1} ^ {angka2} = {hasil}")
    elif pilihan == '7':  # Akar Kuadrat
        hasil = akar_kuadrat(angka1)
        if isinstance(hasil, str):  # Jika hasil adalah pesan error
            print(hasil)
        else:
            print(f"Hasil: √{angka1} = {hasil:.10f}")  # Menampilkan hasil dengan 10 desimal
    
    print()  # Baris kosong untuk memisahkan setiap operasi
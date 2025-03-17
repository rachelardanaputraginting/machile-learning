def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2  # Cari indeks tengah
        
        # Jika target ditemukan
        if arr[mid] == target:
            return mid
        
        # Jika target lebih kecil, cari di bagian kiri
        elif arr[mid] > target:
            right = mid - 1
        
        # Jika target lebih besar, cari di bagian kanan
        else:
            left = mid + 1
    
    return -1  # Mengembalikan -1 jika target tidak ditemukan

# Contoh penggunaan
arr = [10, 20, 30, 50, 60, 80, 100, 110, 130, 170]  # Array harus terurut
target = 110
result = binary_search(arr, target)

if result != -1:
    print(f"Target {target} ditemukan pada indeks {result}")
else:
    print(f"Target {target} tidak ditemukan dalam array")
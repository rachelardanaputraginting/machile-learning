def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Mengembalikan indeks jika target ditemukan
    
    return -1  # Mengembalikan -1 jika target tidak ditemukan

# Contoh penggunaan
arr = [10, 20, 80, 30, 60, 50, 110, 100, 130, 170]
target = 110
result = linear_search(arr, target)

if result != -1:
    print(f"Target {target} ditemukan pada indeks {result}")
else:
    print(f"Target {target} tidak ditemukan dalam array")
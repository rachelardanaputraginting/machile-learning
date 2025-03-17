def binary_search_recursive(arr, target, left=None, right=None):
    """
    Implementasi algoritma Binary Search secara rekursif
    
    Input: array/list TERURUT dan nilai target yang dicari
    Output: indeks posisi target jika ditemukan, -1 jika tidak ditemukan
    
    Kompleksitas Waktu: O(log n)
    """
    # Inisialisasi left dan right jika belum ditentukan
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # Basis rekursi: jika left > right, target tidak ditemukan
    if left > right:
        return -1
    
    # Cari indeks tengah
    mid = (left + right) // 2
    
    # Jika target ditemukan
    if arr[mid] == target:
        return mid
    
    # Jika target lebih kecil, cari di bagian kiri
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    
    # Jika target lebih besar, cari di bagian kanan
    else:
        return binary_search_recursive(arr, target, mid + 1, right)

# Contoh penggunaan
arr = [10, 20, 30, 50, 60, 80, 100, 110, 130, 170]  # Array harus terurut
target = 110
result = binary_search_recursive(arr, target)

if result != -1:
    print(f"Target {target} ditemukan pada indeks {result}")
else:
    print(f"Target {target} tidak ditemukan dalam array")
import math

def jump_search(arr, target):
    """
    Implementasi algoritma Jump Search
    
    Input: array/list TERURUT dan nilai target yang dicari
    Output: indeks posisi target jika ditemukan, -1 jika tidak ditemukan
    
    Kompleksitas Waktu: O(√n)
    """
    n = len(arr)
    
    # Tentukan ukuran lompatan (jump)
    jump = int(math.sqrt(n))
    
    # Temukan blok tempat target mungkin berada
    left = 0
    right = 0
    
    # Lompat maju hingga menemukan blok yang berisi target
    while right < n and arr[right] < target:
        left = right
        right = min(right + jump, n - 1)
    
    # Lakukan pencarian linear di blok yang ditemukan
    for i in range(left, min(right + 1, n)):
        if arr[i] == target:
            return i
    
    return -1  # Mengembalikan -1 jika target tidak ditemukan

# Contoh penggunaan
arr = [10, 20, 30, 50, 60, 80, 100, 110, 130, 170]  # Array harus terurut
target = 110
result = jump_search(arr, target)

if result != -1:
    print(f"Target {target} ditemukan pada indeks {result}")
else:
    print(f"Target {target} tidak ditemukan dalam array")
def interpolation_search(arr, target):
    """
    Implementasi algoritma Interpolation Search
    
    Input: array/list TERURUT dan nilai target yang dicari
    Output: indeks posisi target jika ditemukan, -1 jika tidak ditemukan
    
    Kompleksitas Waktu: O(log log n) untuk distribusi seragam, O(n) untuk kasus terburuk
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        # Rumus interpolasi untuk menentukan posisi
        # Formula: left + ((target - arr[left]) * (right - left)) / (arr[right] - arr[left])
        
        # Hindari pembagian dengan nol
        if arr[left] == arr[right]:
            if arr[left] == target:
                return left
            return -1
        
        pos = left + int(((target - arr[left]) * (right - left)) / (arr[right] - arr[left]))
        
        if arr[pos] == target:
            return pos
        
        if arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1  # Mengembalikan -1 jika target tidak ditemukan

# Contoh penggunaan
arr = [10, 20, 30, 50, 60, 80, 100, 110, 130, 170]  # Array harus terurut
target = 110
result = interpolation_search(arr, target)

if result != -1:
    print(f"Target {target} ditemukan pada indeks {result}")
else:
    print(f"Target {target} tidak ditemukan dalam array")
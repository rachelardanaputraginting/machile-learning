def merge_sort(arr):
    # Buat salinan array agar tidak mengubah array asli
    result = arr.copy()
    
    # Basis rekursi: jika panjang array ≤ 1, kembalikan array tersebut
    if len(result) <= 1:
        return result
    
    # Bagi array menjadi dua bagian
    mid = len(result) // 2
    left_half = result[:mid]
    right_half = result[mid:]
    
    # Urutkan kedua bagian secara rekursif
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Gabungkan kedua bagian yang sudah diurutkan
    return merge(left_half, right_half)

def merge(left, right):
    """
    Fungsi untuk menggabungkan dua array terurut
    """
    result = []
    i = j = 0
    
    # Bandingkan elemen dari kedua array dan gabungkan secara terurut
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Tambahkan elemen yang tersisa (jika ada)
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Contoh penggunaan
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)
print("Array asli:", arr)
print("Array setelah diurutkan:", sorted_arr)
def bubble_sort(arr):
    n = len(arr)
    # Buat salinan array agar tidak mengubah array asli
    result = arr.copy()
    
    for i in range(n):
        # Loop untuk setiap pasangan elemen yang bersebelahan
        for j in range(0, n - i - 1):
            # Bandingkan elemen bersebelahan
            if result[j] > result[j + 1]:
                # Tukar posisi jika elemen tidak berurutan
                result[j], result[j + 1] = result[j + 1], result[j]
    
    return result

# Contoh penggunaan
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print("Array asli:", arr)
print("Array setelah diurutkan:", sorted_arr)
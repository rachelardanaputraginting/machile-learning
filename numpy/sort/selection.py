def selection_sort(arr):
    n = len(arr)
    # Buat salinan array agar tidak mengubah array asli
    result = arr.copy()
    
    for i in range(n):
        # Cari nilai minimum dalam array yang belum diurutkan
        min_index = i
        for j in range(i + 1, n):
            if result[j] < result[min_index]:
                min_index = j
        
        # Tukar elemen minimum dengan elemen pertama
        result[i], result[min_index] = result[min_index], result[i]
    
    return result

# Contoh penggunaan
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = selection_sort(arr)
print("Array asli:", arr)
print("Array setelah diurutkan:", sorted_arr)
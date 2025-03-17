def insertion_sort(arr):
    n = len(arr)
    # Buat salinan array agar tidak mengubah array asli
    result = arr.copy()
    
    for i in range(1, n):
        # Simpan nilai elemen yang akan dimasukkan ke tempat yang tepat
        key = result[i]
        j = i - 1
        
        # Geser elemen yang lebih besar dari key ke posisi satu langkah di depan
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j = j - 1
            
        # Masukkan key ke posisi yang tepat
        result[j + 1] = key
    
    return result

# Contoh penggunaan
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = insertion_sort(arr)
print("Array asli:", arr)
print("Array setelah diurutkan:", sorted_arr)
def quick_sort(arr):
    # Buat salinan array agar tidak mengubah array asli
    result = arr.copy()
    
    # Fungsi rekursif untuk quick sort
    def _quick_sort(arr, low, high):
        if low < high:
            # Pilih pivot dan partisi array
            pivot_index = partition(arr, low, high)
            
            # Urutkan sub-array sebelum dan sesudah pivot
            _quick_sort(arr, low, pivot_index - 1)
            _quick_sort(arr, pivot_index + 1, high)
    
    def partition(arr, low, high):
        # Pilih elemen paling kanan sebagai pivot
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            # Jika elemen saat ini lebih kecil dari pivot
            if arr[j] <= pivot:
                i += 1
                # Tukar posisi
                arr[i], arr[j] = arr[j], arr[i]
        
        # Tempatkan pivot di posisi yang tepat
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    # Panggil fungsi quick sort rekursif
    _quick_sort(result, 0, len(result) - 1)
    return result

# Contoh penggunaan
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)
print("Array asli:", arr)
print("Array setelah diurutkan:", sorted_arr)
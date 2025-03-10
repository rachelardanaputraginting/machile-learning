import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr_3d = arr.reshape(2, 3, 2)
newarr = arr.reshape(3, 4)

print(newarr)
print(newarr_3d)
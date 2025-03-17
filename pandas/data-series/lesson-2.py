import pandas as pd 

a = [1,7,2]
myVar = pd.Series(a, index=["X", "Y", "Z"])

print(myVar);
print(myVar["y"])
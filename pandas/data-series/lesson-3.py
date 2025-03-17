import pandas as pd 

a = {"day1": 420, "day2": 380, "day3": 390}
myVar = pd.Series(a, index=["day1", "day2", "day3"])

print(myVar);
print(myVar["day1"])
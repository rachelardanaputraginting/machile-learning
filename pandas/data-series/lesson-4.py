import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

for i in range(len(df)):
    print(df.loc[i])
    print()

# print(df) 
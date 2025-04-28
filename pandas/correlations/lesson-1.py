import pandas as pd

df = pd.read_csv('data.csv')

df.corr()
print(df.corr())
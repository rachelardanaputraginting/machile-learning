import pandas as pd

df = pd.read_csv('../dataset/data.csv')

df['Date'] = pd.to_datetime(df['Date'], format='mixed')

print(df.to_string())
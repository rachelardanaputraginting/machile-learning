# Increase the maximum number of rows to display the entire DataFrame:
import pandas as pd

pd.options.display.max_rows = 9999

df = pd.read_csv('../dataset/diabetes.csv')

print(df)
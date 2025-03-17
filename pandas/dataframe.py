import pandas as pd

myDataset = {
    'cars': ["BMW", "Volvo", "Ford"],
    'passing': [3, 7, 10],
}

df = pd.DataFrame(myDataset)

print(df);
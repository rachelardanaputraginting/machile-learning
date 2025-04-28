for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120

for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)
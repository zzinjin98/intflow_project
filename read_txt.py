import pandas as pd

df = pd.read_csv("mounting_000_det.txt", sep=",", header=None)
df.columns = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
df = df.drop(index=0, axis=0)

print(df)
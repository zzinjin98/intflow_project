import pandas as pd
from Rotate_function import rotate, rotate_box_dot

df = pd.read_csv("mounting_000_det.txt", sep=",", header=None)
df.columns = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
df = df.drop(index=0, axis=0)

rotate_list = []

for i in range(len(df)):
    data = list(df.loc[1,:])
    result = rotate_box_dot(float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]))
    rotate_list.append(result)

df_rotate = pd.DataFrame(rotate_list, columns=['rotated_x1','rotated_y1', 'rotated_x2','rotated_y2', 'rotated_x3','rotated_y3', 'rotated_x4','rotated_y4'])

print(df_rotate)



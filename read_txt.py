import pandas as pd
import numpy as np
from Rotate_function import rotate, rotate_box_dot

df = pd.read_csv("mounting_001\mounting_001_det.txt", sep=",", header=None)
df.columns = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
df = df.drop(index=0, axis=0)

rotate_list = []

for i in range(len(df)):
    data = list(df.loc[i+1,:])
    result = rotate_box_dot(int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]))
    rotate_list.append(result)

df_rotate = pd.DataFrame(rotate_list, columns=['frame','rotated_x1','rotated_y1', 'rotated_x2','rotated_y2', 'rotated_x3','rotated_y3', 'rotated_x4','rotated_y4'])



df_0 = df_rotate[df_rotate['frame']==1]
df_array_0 = []
# for i in range(len(df_0)):
#     df_array_0.append(np.reshape(list(df_0.loc[i, :])[1:], (4,2)))

print(df_rotate)
print(df_0)
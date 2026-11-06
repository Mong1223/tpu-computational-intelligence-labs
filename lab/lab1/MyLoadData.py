import numpy as np
import pandas as pd

filename = 'diabetes3.dt'
f = open(filename, 'r')
data = []
for line in f:
  data.append(line)
del data[:7]
for i in range(len(data)):
  data[i]=data[i].replace('\n','')
  data[i]=[np.double(x) for x in data[i].split(' ')]
df = pd.DataFrame(data)

X = df.values[:,:8]

Y = df.values[:,9]
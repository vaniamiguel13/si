import sys

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

sys.path.append('D:/Mestrado/2ano/1semestre/SIB/si/datasets')
import pandas as pd
import numpy as np
from dataset import Dataset

def read_csv (filename, sep = ',', features = False, label = False):

    df = pd.read_csv(filename, sep=sep)
    indx = (list(df.columns))
    if features==True:
        ft=indx[1:]
    else: ft=None
    if label == True:
        lb = indx[0]
        y= df[df.columns[0]].to_numpy()
        df = df[df.columns[1:]].to_numpy()
    else:
        lb=None
        y=None
        df = df.to_numpy()

    return Dataset(df, y, ft, lb)

print(read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/exq.csv', ',', True, True))

def write_csv (filename, dataset, sep = ',', features = False, label = False):

    df = pd.DataFrame(data=dataset.X)
    if not (dataset.X is None):
        df.columns=dataset.Features

    if not (dataset.Y is None):
        df.insert(loc=0, column=dataset.Label, value=dataset.Y)

    df.to_csv(filename, index=False)
    return 'Done'


x=np.array([[1,2,3], [3,2,3]])
y=np.array([1,2])
features= ['A', 'B','C']
label= 'y'
dataset= Dataset(X=x, y=y, features= features, label=label)
print(write_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/exq.csv', dataset,',', True, True))
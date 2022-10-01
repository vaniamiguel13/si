import sys

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

sys.path.append('D:/Mestrado/2ano/1semestre/SIB/si/datasets')
import pandas as pd
import numpy as np
from dataset import Dataset
from io import StringIO


def read_data_file (filename, sep = ',',label = False):
    df = np.genfromtxt(filename, delimiter=sep, dtype='unicode')
    if label == True:
        ft= df[0:1, 1:]
        X = df[1:, 1:]
        y = df[1:, 0]
        lb=df[0:1, 0]
    else:
        ft = df[0:1, :]
        lb=None
        X = df[1:, :]
        y = None
    return Dataset(X,y,features=ft, label=lb)
print(read_data_file ('D:/Mestrado/2ano/1semestre/SIB/si/datasets/o.txt', sep = ',',label=True))

def write_data_file (filename, dataset, sep = ',',label = False):

    df = pd.DataFrame(data=dataset.X)
    if not (dataset.X is None):
        df.columns = dataset.Features

    if not (dataset.Y is None):
        df.insert(loc=0, column=dataset.Label, value=dataset.Y)

    df=df.to_numpy()
    col= dataset.Features
    if dataset.Y is not None:
        col.insert(0,dataset.Label)

    np.savetxt(filename, df, delimiter=sep, header=','.join(col), comments='')
    return 'Done'

x=np.array([[1,2,3], [3,2,3]])
y=np.array([1,2])
features= ['A', 'B','C']
label= 'y'
dataset= Dataset(X=x, y=None, features= features, label=None)
print(write_data_file('D:/Mestrado/2ano/1semestre/SIB/si/datasets/o.txt', dataset,',',False))
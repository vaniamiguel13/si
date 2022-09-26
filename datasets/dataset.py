import numpy as np
import pandas as pd
class Dataset:
    def __init__(self, X, y=None, features= None, label= None):
        #x = numy array
        #y = array de 1 dim
        #features = lista ou tuplo
        #label = str
        self.X= X
        self.Y= y
        self.Features = features 
        self.Label = label

    def shape(self):
        return x.shape
    
    def has_label(self):
        if y == None:
            return False
        else:
            return True
    
    def get_classes(self):
        if self.Y is None:
            return 'ERRO'
        return np.unique(self.Y)
    
    def get_mean(self):
        return np.mean(self.X)
    
    def get_mean(self):
        return np.mean(self.X, axis=0)
    
    def get_var(self):
        return np.var(self.X, axis= 0)
    
    def get_median(self):
        return np.median(self.X, axis=0)
    
    def get_min(self):
        return np.min(self.X, axis=0)

    def get_max(self):
        return np.max(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'mean':}
        pass

    
    


if __name__ == '__main__':
    x=np.array([[1,2,3], [3,2,3]])
    y=np.array([1,2])
    features= ['A', 'B','C']        
    label= 'y'
    dataset= Dataset(X=x, y=y, features= features, label=label)
    print(dataset.shape())
    print(dataset.get_classes())
    print(dataset.get_min())







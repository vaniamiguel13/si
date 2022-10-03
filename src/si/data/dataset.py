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


    def __str__(self):
        if not (self.Features is None):
            r=f'X:{str(self.Features)[1:-1]}\n--\n'
        else:
            r = 'X:\n--\n'
        for elem in self.X:
            r += str(elem)[1:-1].replace(' ', '\t') +'\n'
        if not (self.Y is None):
            r+= f'\nY: {self.Label}\n--\n' + str(self.Y).replace(' ', '\t') +'\n'
        return r

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
            {'mean':self.get_mean(),
            'var':self.get_var(),
            'median':self.get_median(),
            'min':self.get_min(),
            'max':self.get_max()})

    def DropNA(self):

        df=pd.DataFrame(self.X, columns= self.Features)

        if not (self.Y is None):
            df.insert(loc=0, column=self.Label, value=self.Y)

        dfn=df.dropna()
        dt=dfn.to_numpy()
        if dt is not None:
            self.Y=[]
            for elem in dt[0:,0:1]:
                self.Y.append(float(elem))
            self.X = dt[0:, 1:]
        return Dataset(self.X, self.Y, self.Features, self.Label)

    def FillNA(self, n_or_m):
        df = pd.DataFrame(self.X, columns=self.Features)
        if not (self.Y is None):
            df.insert(loc=0, column=self.Label, value=self.Y)

        fill_df=df.fillna(n_or_m)
        dt = fill_df.to_numpy()
        if dt is not None:
            self.Y = []
            for elem in dt[0:, 0:1]:
                self.Y.append(float(elem))
            self.X = dt[0:, 1:]
        return Dataset(self.X, self.Y, self.Features, self.Label)

    

if __name__ == '__main__':
    x=np.array([[1,np.nan,3], [3,2,3], [3,2,3]])
    y=np.array([1,2,3])
    features= ['A', 'B','C']        
    label= 'y'
    dataset= Dataset(X=x, y=y, features= features, label=label)
    print(dataset.get_var())

    print(dataset.FillNA(0))








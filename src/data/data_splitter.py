# has tools for model selection
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self,test_size):
        self.test_size= test_size 

    def data_splitter(self,selected_features,Y):
        x_train, x_test, y_train, y_test = train_test_split(selected_features,Y,test_size= self.test_size, shuffle= True)
        return x_train,x_test,y_train,y_test
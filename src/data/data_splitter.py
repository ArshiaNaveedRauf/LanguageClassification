# has tools for model selection
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self,test_size,random_state):
        self.test_size= test_size 
        self.random_state = random_state 

    def data_splitter(self,selected_features,Y):
        x_train, x_test, y_train, y_test = train_test_split(selected_features,Y,test_size= self.test_size, shuffle= True, random_state = self.random_state)
        return x_train,x_test,y_train,y_test
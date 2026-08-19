from abc import ABC, abstractmethod

class BaseClass(ABC):
    def __init__(self,name):
        self.name = name
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def train_model(self,x_train,y_train):
        return self.model.fit(x_train,y_train)


    def predict(self,x):
        return self.model.predict(x)







        
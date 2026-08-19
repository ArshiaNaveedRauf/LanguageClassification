
class ModelTrainer:
    def __init__(self,models):
        self.models= models

    def train_all_models(self,x_train,y_train):
        for model in self.models:
            print(f"training {model.name}")
            model.train_model(x_train,y_train)

        return self.models


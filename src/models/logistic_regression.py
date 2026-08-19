# contains methods for regression
from sklearn.linear_model import LogisticRegression
from src.models.abstract_base_class import BaseClass
from config import maximum_iteration

class LogisticRegressionModel(BaseClass):
    def __init__(self):
        self.max_iterations= maximum_iteration
        super().__init__(name='Logistic Regression')

    def build_model(self):
        return LogisticRegression(max_iter=self.max_iterations)

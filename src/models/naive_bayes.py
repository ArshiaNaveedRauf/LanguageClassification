# applying bayes theorem assuming conditional independence between every pair of features
from sklearn.naive_bayes import MultinomialNB
from src.models.abstract_base_class import BaseClass

class NaiveBayesModel(BaseClass):
    def __init__(self):
        super().__init__(name='Naive Bayes')

    def build_model(self):
        return MultinomialNB()
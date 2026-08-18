# dimentionally reduce sample size
from sklearn.feature_selection import chi2, SelectKBest

class FeatureSelector:
    def __init__(self,k_best_features):
        self.k_best_features= k_best_features
        self.selector = SelectKBest(score_func=chi2,  k=self.k_best_features)

    def feature_selector(self,Y,features): 
        featuresSelected= self.selector.fit_transform(features, Y)
        print(featuresSelected.shape)
        return featuresSelected
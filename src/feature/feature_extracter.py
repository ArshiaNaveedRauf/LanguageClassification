# feature extraction from raw data in a format that is supported by Ml algorithms 
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        self.vectorizer= TfidfVectorizer()

    def feature_extractor(self,X):
        # raw text into a numerical matrix
        features= self.vectorizer.fit_transform(X)
        return features
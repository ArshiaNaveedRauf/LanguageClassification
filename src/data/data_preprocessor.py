# transforms raw data into clean, normlaized form
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self,text_column,language_column,encoded_language_column):
        self.text_column= text_column
        self.language_column= language_column 
        self.encoded_language_column= encoded_language_column
        self.encoder= LabelEncoder()

    def lowercase(self, data):
        data[self.text_column] = data[self.text_column].str.lower()
        return data

    def encoded_language(self,data):
        # fits label encoder and returns encoded label
        data[self.encoded_language_column]= self.encoder.fit_transform(data[self.language_column])
        return data

    def preprocess(self,data):
        data= self.lowercase(data)
        data= self.encoded_language(data)
        return data

        
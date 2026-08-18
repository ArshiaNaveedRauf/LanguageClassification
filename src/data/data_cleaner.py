
class DataCleaner:
    def __init__(self, text_column, language_column):
        self.text_column= text_column
        self.language_column= language_column
        
    def drop_duplicates(self,data):
        data = data.drop_duplicates()
        print (f"Duplicates after droping: {data.duplicated().sum()}")
        return data

    def drop_null(self,data):
        data= data.dropna(subset=[self.text_column, self.language_column])
        return data

    def clean(self,data):
        data = self.drop_duplicates(data)
        data= self.drop_null(data)
        return data
        
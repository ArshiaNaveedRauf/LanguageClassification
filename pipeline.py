from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_preprocessor import DataProcessor
from config import text_column, language_column
from config import encoded_language_column

class LanguageIdentificationPipeline:
    def __init__(self):
        self.loader= DataLoader()
        self.cleaner= DataCleaner(text_column,language_column)
        self.preprocessor=  DataProcessor(text_column,language_column,encoded_language_column)

    def run_pipeline(self):
        data = self.loader.load()
        data= self.cleaner.clean(data)
        data = self.preprocessor.preprocess(data)
        print(data.head())

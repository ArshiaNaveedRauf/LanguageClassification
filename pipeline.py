from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_preprocessor import DataProcessor
from src.eda.data_explorer import DataExplorer
from src.eda.visualizer import Visualizer
from config import text_column, language_column
from config import encoded_language_column

class LanguageIdentificationPipeline:
    def __init__(self):
        self.loader= DataLoader()
        self.cleaner= DataCleaner(text_column,language_column)
        self.preprocessor=  DataProcessor(text_column,language_column,encoded_language_column)
        self.explorer= DataExplorer(language_column)
        self.visualizer= Visualizer(language_column, text_column)

    def run_pipeline(self):
        data = self.loader.load()
        self.explorer.summary(data)
        data= self.cleaner.clean(data)
        data = self.preprocessor.preprocess(data)
        data.to_csv("data/processed/cleaned_data.csv", index=False)
        print(data.head())
        self.visualizer.bar_chart(data)
        self.visualizer.top_words_per_language(data)
        


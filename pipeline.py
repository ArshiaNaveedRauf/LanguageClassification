from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_preprocessor import DataProcessor
from src.eda.data_explorer import DataExplorer
from src.eda.visualizer import Visualizer
from src.feature.feature_extracter import FeatureExtractor
from src.feature.feature_selector import FeatureSelector
from src.data.data_splitter import DataSplitter
from src.models.model_trainer import ModelTrainer
from src.models.logistic_regression import LogisticRegressionModel
from src.models.naive_bayes import NaiveBayesModel
from src.evaluation.model_evaluator import ModelEvaluator
from src.models.model_persistence import ModelPersistence
from config import text_column, language_column
from config import encoded_language_column
from config import k_best_features
from config import test_size
from config import random_state

class LanguageIdentificationPipeline:
    def __init__(self):
        self.loader= DataLoader()
        self.cleaner= DataCleaner(text_column,language_column)
        self.preprocessor=  DataProcessor(text_column,language_column,encoded_language_column)
        self.explorer= DataExplorer(language_column)
        self.visualizer= Visualizer(language_column, text_column)
        self.extractor = FeatureExtractor()
        self.selector = FeatureSelector(k_best_features)
        self.splitter= DataSplitter(test_size,random_state)
        self.models=[LogisticRegressionModel(), NaiveBayesModel()]
        self.trainer = ModelTrainer(self.models)
        self.evaluator= ModelEvaluator(self.models)
        self.persistence= ModelPersistence()
    



    def run_pipeline(self):
        data = self.loader.load()
        self.explorer.summary(data)
        data= self.cleaner.clean(data)
        data = self.preprocessor.preprocess(data)
        data.to_csv("dataset/processed/cleaned_data.csv", index=False)
        print(data.head())
        self.visualizer.bar_chart(data)
        self.visualizer.top_words_per_language(data)
        X = data[text_column]
        Y = data[encoded_language_column]
        features = self.extractor.feature_extractor(X)
        selected_features= self.selector.feature_selector(Y,features)
        x_train,x_test,y_train,y_test= self.splitter.data_splitter(selected_features,Y)
        self.trainer.train_all_models(x_train,y_train)
        self.evaluator.evaluation(y_test, x_test)
        best_model_name=  self.evaluator.model_comparision()

    # save the model
        best_model_obj= next(m for m in self.models if m.name== best_model_name )
        self.persistence.save (best_model_obj, "models_joblib/best_model.joblib")
        self.persistence.save(self.extractor.vectorizer,"models_joblib/vectorizer.joblib")
        self.persistence.save(self.selector.selector, "models_joblib/selector.joblib")
        self.persistence.save(self.preprocessor.encoder,"models_joblib/encoder.joblib")





        


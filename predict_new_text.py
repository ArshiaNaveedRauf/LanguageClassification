# predict.py (new file, project root)
from src.models.model_persistence import ModelPersistence
from prediction import LanguagePredictor

persistence = ModelPersistence()
model = persistence.load("models_joblib/best_model.joblib")
vectorizer = persistence.load("models_joblib/vectorizer.joblib")
selector = persistence.load("models_joblib/selector.joblib")
encoder = persistence.load("models_joblib/encoder.joblib")

predictor = LanguagePredictor(model, vectorizer, selector, encoder)

result = predictor.predict("bonjour comment allez vous")
print(result)
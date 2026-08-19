class LanguagePredictor:
    def __init__(self,model, vectorizer, selector, encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.selector = selector
        self.encoder = encoder

    def predict(self,text):
        text = text.lower()
        text_vector= self.vectorizer.transform([text])
        selected_features = self.selector.transform(text_vector)
        prediction = self.model.predict(selected_features)
        language = self.encoder.inverse_transform(prediction)
        return language
        

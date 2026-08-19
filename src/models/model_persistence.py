import joblib

class ModelPersistence:
    def save(self, obj, path):
        joblib.dump(obj,path)
        print("Model Saved")

    def load(self, path):
        return joblib.load(path)


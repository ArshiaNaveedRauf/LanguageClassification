from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self,models):
        self.models =models

    def accuracy_calculation(self,y_test,model_name,predictions):
            print(f" Accuracy Score for {model_name}: {accuracy_score(y_test,predictions)}")

    def classification_results(self,y_test,model_name,predictions):
            print(f" Classification Report of {model_name}: {classification_report(y_test,predictions)}")


    def confusion_matrix_output(self, y_test, model_name,predictions):
            ConfusionMatrixDisplay.from_predictions(y_test,predictions)
            plt.title(f"confusion matrix {model_name}")
            plt.savefig(f"outputs/figures/confusion_matrix_{model_name}.png")
            plt.close()


    def evaluation(self, y_test, x_test):
        for model in self.models:
            model_name= model.name
            predictions= model.predict(x_test)
            self.accuracy_calculation( y_test,model_name,predictions)
            self.classification_results( y_test,model_name,predictions)
            self.confusion_matrix_output( y_test,model_name,predictions)


        
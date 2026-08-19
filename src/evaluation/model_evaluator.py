from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self,models):
        self.models =models
        self.accuracy_score_dict = {}

    def accuracy_calculation(self,y_test,model_name,predictions):
            accuracy = accuracy_score(y_test,predictions)
            print(f" Accuracy Score for {model_name}: {accuracy}")
            self.accuracy_score_dict[model_name] = accuracy

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

    def model_comparision(self):
          best_model= max(self.accuracy_score_dict,key= self.accuracy_score_dict.get)
          better_accuracy=self.accuracy_score_dict[best_model]
          print(f"{best_model} has better accuracy score: {better_accuracy*100: .2f}%")
          return best_model


        
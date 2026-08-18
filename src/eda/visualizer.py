import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
# changes words into numbers
from sklearn.feature_extraction.text import CountVectorizer

class Visualizer:
    def __init__(self,language_column,text_column):
        self.CountVectorizer=CountVectorizer()
        self.language_column= language_column
        self.text_column= text_column

    def bar_chart(self,data):
        data[self.language_column].value_counts().plot(kind="bar", figsize=(10,5))
        plt.xlabel("Languages")
        plt.ylabel("Number Of Samples")
        plt.title("Language Distribution")
        plt.savefig("outputs/figures/language_distribution.png")
        plt.close()

    def top_words_per_language(self,data):
        # get every unique language in the dataset
        languages= data[self.language_column].unique()
        # loop that runs for every unique language
        for language in languages:
        # get all the rows for that paticular language
            texts = data[data[self.language_column]== language][self.text_column]
            # count the frequency of each word
            Vectorizer= self.CountVectorizer
            # find all words and convert into a matrix of word counts 
            TransformedWords=  Vectorizer.fit_transform(texts)
            # adds up the value of each word across all documents 
            wordCounts= TransformedWords.sum(axis=0).A1
            # names of the words 
            word = Vectorizer.get_feature_names_out()
            # sort the pair and get top 10
            topWords = sorted(
                zip(word, wordCounts),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            # sep the pair 
            words, counts = zip(*topWords)
            # make dataframe 
            dataTopWords= pd.DataFrame(topWords, columns=['word','count'])
            sns.barplot(data=dataTopWords, x='count', y='word')
            plt.title(f'Top 10 Words in {language}')
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            plt.savefig(f"outputs/figures/top_words_{language}.png")
            plt.close()
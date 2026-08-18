# checks raw data
bold= '\033[1m'
end='\033[0m'

class DataExplorer:
    def __init__(self,language_column):
        self.language_column=language_column
        

    def summary(self,data):
        print(f"{bold}first 5 Rows:{end} \n{data.head(5)}\n")
        # printing information about the data
        print (f"{bold} {data.info()} \n")
        # number of duplicates 
        print(f"{bold}Number of Duplicates:{end} {data.duplicated().sum()} \n")
        # checking null values 
        print(f"{bold}Number of Null values:{end} {data.isnull().sum()} \n")
        # printing labels
        print(f"{bold}Columns: {data.columns}\n")
        # printing unique languages 
        print(f"{bold}languages:{end} \n {data[self.language_column].unique()} \n")
        # count of each unique value
        print(f"{bold}Frequency of Unique languages:{end} \n {data[self.language_column].value_counts()} \n")


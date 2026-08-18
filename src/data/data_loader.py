import pandas as pd 
from pathlib import Path
from config import dataset_path

class DataLoader:
    def __init__(self):
        self.filename = dataset_path

    def load(self):
        if self.filename == "":
            print("dataset does not exist")

        return pd.read_csv(self.filename)

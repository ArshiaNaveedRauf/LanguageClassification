from src.data.data_cleaner import DataCleaner
import pandas as pd

def test_remove_duplicated_rows():
    data = pd.DataFrame({"Text": ["hello", "hello","bonjour madam"], "language":["English","English","French"]})
    cleaner= DataCleaner("Text","language")
    result= cleaner.drop_duplicates(data)
    assert len(result) ==2

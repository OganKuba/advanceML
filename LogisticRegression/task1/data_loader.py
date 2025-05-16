import pandas as pd

def load_earthquake_data(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", header=0)  # dopasuj sep do swojego pliku
    return df
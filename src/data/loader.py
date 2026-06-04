
import pandas as pd
import os

def load_csv(file_name):
    return pd.read_csv(os.path.join("data/raw", file_name))

import pandas as pd
from pathlib import Path


def write_csv(df, file):
    if Path(file).exists():
        df.to_csv(file, index=0, mode='a', header=False)
    else:
        df.to_csv(file, index=0)


def read_csv(file):
    if not Path(file).exists():
        raise FileNotFoundError('No such .csv file: ' + str(file))
    else:
        df = pd.read_csv(file)
        return df

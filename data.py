from pathlib import Path

import pandas as pd

DEPENDENT = 9


def independents(df):
    return [int(c) for c in df.columns if c != DEPENDENT]


def read_csv(path=None, **csv_kwargs):
    if path is None:
        path = 'input/train.txt.gz'
    if not Path(path).exists():
        msg_fmt = 'Please download the data set at {} and save to {}'
        url = 'https://drive.google.com/open?id=0BwpUS-D5xBA9WnlMYU5sSkRHUmM'
        print(msg_fmt.format(url, path))
    return pd.read_csv(path, names=range(10), **csv_kwargs)

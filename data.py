from pathlib import Path

import pandas as pd

DEPENDENT = 9
INDEPENDENTS = list(range(8))


def read_csv(path=None):
    # 35 seconds to extract all data on my laptop, ~500MB used by dataframe
    if path is None:
        path = 'input/train.txt.gz'
    if not Path(path).exists():
        # TODO(rheineke): Automagically download. Would require authentication
        # https://developers.google.com/drive/v3/web/quickstart/python
        msg_fmt = 'Please download the data set at {} and save to {}'
        url = 'https://drive.google.com/open?id=0BwpUS-D5xBA9WnlMYU5sSkRHUmM'
        print(msg_fmt.format(url, path))
    kwargs = dict(
        index_col=8,
        names=range(10),
        nrows=None,  # 100000
    )
    return pd.read_csv(path, **kwargs)

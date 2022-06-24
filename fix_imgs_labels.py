import pandas as pd
import os
from pathlib import Path
import warnings


def rearrange(dir, img_name, op):
    file = os.path.join(dir, img_name)
    if op == "delete":
        if os.path.isfile(file):
            os.remove(file)
        else:
            warnings.warn(f"couldn't find file: {file}")
    else:
        Path(op).mkdir(parents=True, exist_ok=True)
        dest = os.path.join(op, img_name)
        if os.path.isfile(file):
            os.rename(file, dest)
        else:
            warnings.warn(f"couldn't find file: {file}")


if __name__ == "__main__":
    df = pd.read_csv('./fixes.csv')
    for _, row in df.iterrows():
        rearrange(row['path'], row['img'], row['dest'])

# Raw dataset

The raw CSV is downloaded from:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

Run `python src/preprocess.py` from the project root to download and save it as `data/raw/titanic.csv`.
The script does not silently replace an existing raw file unless download fails or you remove the file first.

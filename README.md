# Virtual-Data-Science-with-Python-Trainee-Week-1-Task
This my week 1 task of Virtual Data Science with Python Trainee intenship of yuva intern, in this I learned about Data Acquisition, Cleaning, and Preprocessing.

**Project:** Titanic passenger dataset  
**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn  
**Purpose:** Demonstrate a reproducible data-cleaning workflow for a public dataset.

## Dataset
This project uses the public Titanic dataset maintained in the `seaborn-data` repository:
- Source CSV: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
- Repository: https://github.com/mwaskom/seaborn-data
- Dataset description: https://github.com/mwaskom/seaborn-data/blob/master/README.md

The script downloads the dataset at runtime and saves a local raw copy. An internet connection is needed for the first run. The raw file is never overwritten by the cleaning script.

## Quick start
1. Install Python 3.10 or later.
2. Open a terminal in the project folder.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the workflow:
   ```bash
   python src/preprocess.py
   ```
5. Review:
   - `data/processed/titanic_cleaned.csv`
   - `data/processed/quality_summary.json`
   - charts in `images/`
   - generated text report in `report/run_summary.txt`

You can also open `notebooks/data_cleaning_walkthrough.ipynb` in Jupyter or Google Colab.

## Methodological choices
- Missing numeric values are imputed with the median; categorical values use the mode or an explicit `"Unknown"` category, depending on the column.
- Exact duplicate rows are reported and removed only if present.
- Outliers are identified with the IQR rule and reported, not automatically deleted. Unusual fares or ages can be valid observations.
- Impossible values are checked with domain rules. The script does not invent a survival label or remove rows simply because they look unusual.
- `survived` is retained as a target column for educational analysis. For predictive modeling, imputation/scaling should be fitted on training data only to avoid data leakage.

## Reproducibility note
The exact output depends on the dataset version available at the source and the installed library versions. The script writes the actual counts from the downloaded file to `quality_summary.json`; use those generated results when updating the report.

## Repository contents
- `src/preprocess.py`: end-to-end reproducible cleaning pipeline
- `notebooks/data_cleaning_walkthrough.ipynb`: guided notebook
- `report/Week1_Data_Preprocessing_Report.docx`: report template with methodology and illustrative baseline notes
- `data/raw/README.md`: raw-data provenance
- `data/processed/`: generated outputs (created when script runs)
- `images/`: generated visualizations

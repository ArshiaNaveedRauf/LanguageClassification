## Setup

1. **Create and activate a virtual environment** (project requirement):

   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate

2. **Install dependencies:**

   pip install -r requirements.txt


3. **Add your dataset:**

   Place your `dataset.csv` (with columns `Text` and `language`) into
   `dataset/raw/dataset.csv`.

## Usage

### Train and evaluate all models

```bash
python3 main.py
```
This runs the full pipeline: load -> explore -> clean -> lowercase -> encodes -> EDA
-> feature engineering -> feature selection -> split -> train (Logistic
Regression + Naive Bayes) -> evaluate -> compare -> save. Plots land in
`outputs/figures/`, cleaned data in `data/processed/`, and trained
artifacts in `models/`.

**Known Limitations**
**Logistic Regression:** 95.27%
**Naive Bayes:** 95%

model currently misclassifies some languages, likely due to limited training data/class imbalance / needs hyperparameter tuning


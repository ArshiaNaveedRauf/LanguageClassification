## Setup

1. **Create and activate a virtual environment** (project requirement):

   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate

2. **Install dependencies:**

   pip install -r requirements.txt


3. **Add your dataset:**

   Place your `dataset.csv` (with columns `Text` and `language`) into
   `data/raw/dataset.csv`. If your file has different column names or
   lives elsewhere, update `PathConfig` / `DataConfig` in `config.py`.

## Usage

### Train and evaluate all models

```bash
python main.py
```

This runs the full pipeline: load → understand → clean → preprocess → EDA
→ feature engineering → feature selection → split → train (Logistic
Regression + Naive Bayes) → evaluate → compare → save. Plots land in
`outputs/figures/`, cleaned data in `data/processed/`, and trained
artifacts in `models/`.

### Predict the language of new text

```bash
python scripts/predict.py "bonjour madame" "hello there" "hola amigo"
```

## Extending the Project

- **Add a new model:** create a new class in `src/models/` that subclasses
  `BaseClassifier` and implements `_build_model()`, then add it to
  `LanguageIdentificationPipeline._get_candidate_models()`.
- **Hyperparameter tuning:** wrap any `BaseClassifier`'s `.model` with
  `sklearn.model_selection.GridSearchCV`/`RandomizedSearchCV` inside
  `ModelTrainer`, or add a dedicated `HyperparameterTuner` class following
  the same single-responsibility pattern.
- **New features:** add a class to `src/features/` following the
  `fit_transform` / `transform` contract used by `TfidfFeatureExtractor`.

## Notes on the Original Notebook

The original notebook (`LanguageIdentification.ipynb`) mixed data
loading, cleaning, feature engineering, training, and evaluation into
loosely related top-level functions, with hard-coded absolute file paths
and undefined variables in the inference cell (`new_text_tfidf` was never
defined). This refactor fixes those bugs and re-organizes the same logic
into the OOP structure described above.
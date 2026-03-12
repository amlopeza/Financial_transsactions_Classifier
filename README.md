# Financial Transaction Classification

## Problem

Classifying financial transactions into categories is essential for personal finance management, fraud detection, and automated bookkeeping. This project tackles a **multi-class text classification** task: given a short transaction description (e.g., *"Uber ride INR 48648 TXNde8842f7"*), predict which of the following **9 categories** it belongs to:

`education` | `entertainment` | `emi` | `food` | `healthcare` | `investment` | `shopping` | `travel` | `utilities`

## Data

| Split | Samples | Source |
|-------|---------|--------|
| Train | 5 000   | `data/raw/train_transactions.csv` |
| Test  | 1 000   | `data/raw/test_transactions.csv`  |

Each record contains two columns:

- **`transaction_text`** -- short description of the transaction (27-48 characters).
- **`category`** -- target label (one of the 9 classes listed above).

The dataset is **balanced** (530-602 samples per class), has **no missing values**, and contains **no duplicate** transaction texts.

## Metrics

| Metric | Purpose |
|--------|---------|
| **F1 Macro** | Primary metric used for hyperparameter tuning (GridSearchCV). Chosen because it weighs all classes equally regardless of support. |
| Accuracy | Overall correctness. |
| Precision / Recall | Per-class and weighted averages reported via `classification_report`. |

## Model Plan

The classification pipeline has three stages:

1. **Custom Text Preprocessor**
   - Lowercases text.
   - Replaces transaction codes (`TXN...`) with a `txn_code` token.
   - Replaces currency symbols (`INR`, `USD`, etc.) with a `currency` token.
   - Replaces numeric amounts with an `amount` token.
   - Optionally removes English stopwords (via NLTK).

2. **TF-IDF Vectorizer** -- converts preprocessed text into numerical features (up to 5 000 features, unigrams or bigrams).

3. **Logistic Regression** -- linear classifier with tunable regularization.

**Hyperparameter tuning** is performed with `GridSearchCV` (5-fold CV, scored on F1 Macro) over:

| Parameter | Values explored |
|-----------|-----------------|
| `preprocessor__remove_stopwords` | `True`, `False` |
| `vectorizer__ngram_range` | `(1,1)`, `(1,2)` |
| `classifier__C` | `0.1`, `1`, `10` |

Best configuration found: `remove_stopwords=True`, `ngram_range=(1,1)`, `C=0.1`.

## Results & Limitations

The model achieves **100% accuracy** and **1.00 F1 Macro** on the test set. No data leakage was found — the train/test split contains no duplicate texts and preprocessing is fitted only on training data.

This perfect score indicates that the **dataset is too simple**: each category contains highly distinctive keywords (e.g., "Uber ride" → travel, "Netflix" → entertainment) that make the classification task trivial for a linear model. The results should not be interpreted as evidence of a robust production-ready classifier. A more challenging, real-world dataset with ambiguous or overlapping descriptions would be needed to properly benchmark the pipeline.

## How to Run

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/clf_Financial_transacction.git
cd clf_Financial_transacction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install scikit-learn
```

### Run the notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebook walks through:

1. **Exploratory Data Analysis** -- class distribution, text length statistics, word clouds.
2. **Preprocessing & Pipeline** -- building the custom transformer and sklearn pipeline.
3. **Hyperparameter Tuning** -- grid search with cross-validation.
4. **Evaluation** -- classification report and performance metrics on the test set.

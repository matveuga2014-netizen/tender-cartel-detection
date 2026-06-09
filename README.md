# Cartel Detection in Public Procurement

Machine learning for detecting bid-rigging cartels among suppliers in Russian
public tenders (44-FZ and 223-FZ). The project covers the full cycle: from
scraping data off the official procurement portal to training and comparing
gradient boosting models.

## Overview

Bid-rigging cartels — where suppliers secretly agree in advance on who will win
while faking competition — cause direct losses to the public budget. The goal of
this project is to build a model that, given contract features and bidder
behavior, estimates the probability of collusion to help prioritize audits.

**Task:** binary classification (collusion / no collusion) on highly imbalanced
data.

## Results

Dataset: **18,537 contracts**, collusion rate **7.9%** (1,471 cases).

CatBoost and LightGBM were compared under two validation schemes. `GroupKFold`
(split by supplier tax ID, INN) tests generalization to **previously unseen
suppliers** — a fair estimate for real-world use.

| Model | Validation | PR-AUC | ROC-AUC | F1 |
|-------|------------|:------:|:-------:|:----:|
| LightGBM | StratifiedKFold | **0.778** | 0.965 | 0.679 |
| CatBoost | StratifiedKFold | 0.731 | 0.959 | 0.587 |
| CatBoost | GroupKFold (by INN) | 0.531 | 0.899 | 0.497 |
| LightGBM | GroupKFold (by INN) | 0.509 | 0.882 | 0.493 |

Baseline (share of the positive class) is PR-AUC **0.079**. On unseen suppliers
the model beats the baseline by **~5x**.

**Key findings:**

- Under a time-based split (`Time-split`), PR-AUC reaches **0.66–0.68** — the
  model holds up on future contracts.
- The most important feature is `price_drop_pct` (the percentage price reduction
  during the auction): an abnormally small drop often signals faked competition.
- The signing year (`sign_year`) adds **+0.10 to PR-AUC** — collusion patterns
  shift over time.

## Pipeline

The project consists of three Jupyter notebooks reflecting the stages of work:

1. **`Parser1.ipynb`** — data collection. Scrapes contract cards from
   `zakupki.gov.ru`, enriches them with supplier data via an external API, then
   merges the records and normalizes tax IDs (INN).
2. **`Feature_engeneering1.ipynb`** — cleaning and feature construction. Raw
   contracts are turned into **61 features** (17 categorical): price dynamics,
   auction parameters, bidder characteristics, and so on.
3. **`Models_termpaper1.ipynb`** — training and evaluation. CatBoost and
   LightGBM, several validation schemes (Stratified / Group / Time-split),
   feature importance analysis, and a comparison of sampling strategies.

## Repository structure

```
.
├── Parser1.ipynb                 # Stage 1: data collection
├── Feature_engeneering1.ipynb    # Stage 2: feature engineering
├── Models_termpaper1.ipynb       # Stage 3: models and evaluation
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

> **Note on data.** Large data files (`features.csv` ~118 MB,
> `polniy_pochti.xlsx` ~26 MB) are not included in the repository due to
> GitHub's 100 MB file-size limit. They can be reproduced by running the
> notebooks in order, or provided separately on request.

## Tech stack

Python · pandas · NumPy · scikit-learn · **CatBoost** · **LightGBM** ·
BeautifulSoup · matplotlib · seaborn · Jupyter

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebooks
jupyter notebook
```

Run the notebooks in order: parser → features → models. The parser requires your
own API key — in the code it has been replaced with the placeholder
`YOUR_API_KEY`.

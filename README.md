# linear_regression

A simple linear regression that predicts a car's price from its mileage using gradient descent.

Full documentation:

[![Interactive guide](https://img.shields.io/badge/Interactive_guide-open-2563eb?style=for-the-badge)](https://mtellami.github.io/Linear-regression/)

## Requirements

- Python 3.10 or newer
- Optional: `python3 -m pip install matplotlib` for plotting

## Usage

```bash
python3 src/train.py
python3 src/predict.py
python3 src/evaluate.py
```

`train.py` writes `data/model.json`, `predict.py` prompts for a mileage, and `evaluate.py` prints MAE, RMSE, and R2. Add `--plot` or `--plot-output PATH` to `train.py` to visualize the data and fitted line.

CSV data must contain at least two rows with numeric `km` and `price` columns.
Supported CSV ranges are 0–1,000,000 km and 0–10,000,000 for price, inclusive.
These application limits also reject NaN and infinity. The limits are defined as
`MAX_MILEAGE` and `MAX_PRICE` in `src/linear_regression.py`.

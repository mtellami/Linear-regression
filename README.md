# linear_regression

A simple linear regression that predicts a car's price from its mileage using gradient descent.

Full documentation:

<a href="https://mtellami.github.io/Linear-regression/" style="display:inline-block;padding:14px 28px;background-color:#2563eb;color:#ffffff;text-decoration:none;border-radius:8px;font-size:18px;font-weight:700;">Interactive guide</a>

## Requirements

- Python 3.10 or newer
- Optional: `python3 -m pip install matplotlib` for plotting

## Usage

```bash
python3 src/train.py
python3 src/predict.py
python3 src/evaluate.py
```

`train.py` writes `data/model.json`, `predict.py` prompts for a mileage, and `evaluate.py` prints MAE, RMSE, and R2. Add `--plot` or `--plot-output PATH` to visualize the data and fitted line. Run any command with `--help` for all options.

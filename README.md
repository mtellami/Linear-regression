# linear_regression

A single-feature linear regression that estimates a car's price from its mileage using gradient descent. The core implementation uses only the Python standard library; Matplotlib is optional and needed only for plots.

## Requirements

- Python 3.10 or newer
- Optional: `python3 -m pip install matplotlib` for plotting

## Quick start

Run the programs from any directory; their default paths are anchored to the project root.

```bash
python3 src/train.py
python3 src/predict.py
python3 src/evaluate.py
```

`train.py` reads `data/data.csv`, trains the model, and writes `data/model.json`. `predict.py` prompts once for a mileage and prints the estimated price. `evaluate.py` reports fit metrics using the saved model.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/linear_regression.py` | Shared CSV loading, validation, model persistence, regression math, and metrics |
| `src/train.py` | Training CLI and plotting |
| `src/predict.py` | Prediction CLI |
| `src/evaluate.py` | Precision-evaluation CLI |
| `data/data.csv` | Supplied `km,price` dataset |
| `data/model.json` | Generated parameters (ignored by Git) |
| `requirements.pdf` | Subject specification |

`README.html` contains an interactive explanatory guide to the same concepts.

## Commands and options

```bash
python3 src/train.py --data data/data.csv --model data/model.json --learning-rate 0.01 --max-iterations 100000
python3 src/train.py --plot
python3 src/train.py --plot-output plot.png
python3 src/predict.py --model data/model.json
python3 src/evaluate.py --data data/data.csv --model data/model.json
```

Relative paths you pass explicitly are resolved from the current working directory. The trainer defaults to learning rate `0.01`, `100000` iterations, and a relative loss-improvement tolerance of `1e-10`; it prints convergence status, iteration count, final mean squared error, and the save location, and warns on stderr when the iteration limit is reached. `--plot` displays the data and fitted line; `--plot-output PATH` saves the same graph. Neither option imports Matplotlib for ordinary training or prediction.

## Model and hypothesis

`data/model.json` contains two numbers:

```json
{
  "theta0": 8499.5377,
  "theta1": -0.021449
}
```

The estimate is `estimatePrice(mileage) = theta0 + theta1 * mileage`. If the model file is missing, both parameters default to zero, so an untrained model predicts `0.00`. Malformed or incomplete model files are errors and are never silently replaced with zeros.

## Training formulas

Mileage is standardized with the population mean and standard deviation:

`z = (km - mean) / std`, where `std = sqrt(sum((km - mean)^2) / m)`.

Training starts with both standardized coefficients at zero. Each iteration computes both gradients from the same current coefficients:

```
tmp0 = learningRate * (1/m) * sum(estimatePrice(z[i]) - price[i])
tmp1 = learningRate * (1/m) * sum((estimatePrice(z[i]) - price[i]) * z[i])
theta0 = theta0 - tmp0
theta1 = theta1 - tmp1
```

Both assignments use the old coefficients, so neither update affects the other within the same iteration. Standardization keeps gradients well-scaled while prices stay in their original units. After training, coefficients are converted back to raw mileage units:

```
theta1 = standardized_slope / std
theta0 = standardized_intercept - theta1 * mean
```

Training stops when the decrease in mean squared error is negligible: `0 <= improvement < 1e-10 * max(previous_loss, 1)`. It also stops at the iteration cap (reported as not converged) or fails immediately if coefficients or loss become non-finite. Failures leave any existing model untouched; successful saves are atomic.

## Validation

- The CSV needs `km` and `price` columns, at least two valid rows, and some mileage variation.
- Empty, nonnumeric, negative, and nonfinite CSV values are rejected with the offending row reported.
- Prediction requires nonnegative, finite mileage; invalid input is rejected rather than re-prompted.
- Training errors (including numerical failure) never overwrite an existing model.

## Metrics

`evaluate.py` reports:

- **MAE**: mean absolute error, in currency units
- **RMSE**: root mean squared error, in currency units, weighting large misses more heavily
- **R2**: coefficient of determination; `1` is a perfect fit, `0` matches a constant-mean predictor, negative values are worse, and it is undefined when target prices are constant

Evaluating against the training CSV measures fit on that dataset, not accuracy on unseen cars. Estimates far outside the training mileage range are extrapolations and can be negative; that sign is preserved in the output.

## AI assistance

AI tools assisted with drafting this implementation and documentation. Design choices were reviewed against the project specification and verified by running the programs; all commands, outputs, and trade-offs remain the responsibility of the project authors.

import sys
from pathlib import Path

from linear_regression import evaluate, load_data, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"


def main() -> int:
    """Load data and model, then print MAE, RMSE, and R2."""
    try:
        mileages, prices = load_data(str(DATA_PATH))
        model = load_model(str(MODEL_PATH))
        mae, rmse, r2 = evaluate(mileages, prices, model)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    if r2 is None:
        print("R2   = undefined (target prices are constant)")
    else:
        print(f"R2   = {r2:.6f}")
    print(
        f"Evaluated on {len(prices)} rows from {DATA_PATH}: "
        "this measures fit on that dataset."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

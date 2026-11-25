import math
import sys
from pathlib import Path

from linear_regression import Model, load_data, load_model, predict_price

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"


def evaluate(
    mileages: list[float], prices: list[float], model: Model
) -> tuple[float, float, float | None]:
    """Compute MAE and RMSE, and R2 unless prices are constant."""
    if len(mileages) != len(prices) or not mileages:
        raise ValueError("evaluation needs matching mileage and price values")
    errors = [
        predict_price(model, mileage) - price
        for mileage, price in zip(mileages, prices)
    ]
    size = len(errors)
    squared_error = sum(error * error for error in errors)
    mean_price = sum(prices) / size
    total_variance = sum((price - mean_price) ** 2 for price in prices)
    r2: float | None = (
        None if total_variance == 0.0 else 1.0 - squared_error / total_variance
    )
    mae = sum(abs(error) for error in errors) / size
    rmse = math.sqrt(squared_error / size)
    return mae, rmse, r2


def main() -> int:
    """Load data and model, then print MAE, RMSE, and R2."""
    try:
        mileages, prices = load_data(str(DATA_PATH))
        model = load_model(str(MODEL_PATH))
        mae, rmse, r2 = evaluate(mileages, prices, model)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    mean_price = sum(prices) / len(prices)
    baseline_rmse = math.sqrt(
        sum((price - mean_price) ** 2 for price in prices) / len(prices)
    )
    print(f"MAE (Mean Absolute Error) = {mae:.2f}")
    print(f"  Predictions miss the actual price by {mae:.2f} price units on average.")
    if mean_price > 0:
        print(
            f"  That is {mae / mean_price:.1%} of the dataset's average price "
            f"({mean_price:.2f})."
        )
    print()
    print(f"RMSE (Root Mean Squared Error) = {rmse:.2f}")
    if baseline_rmse > 0:
        change = (rmse - baseline_rmse) / baseline_rmse
        direction = "lower" if change < 0 else "higher"
        if change == 0:
            print("  This equals the RMSE from always predicting the average price.")
        else:
            print(
                f"  This is {abs(change):.1%} {direction} than the RMSE from "
                f"always predicting the average price ({baseline_rmse:.2f})."
            )
    else:
        print(
            f"  All actual prices are {mean_price:.2f}; always predicting that "
            f"price would give zero RMSE, compared with this model's {rmse:.2f}."
        )
    print()
    if r2 is None:
        print("R2 (Coefficient of Determination) = undefined (target prices are constant)")
        print(
            "  All actual prices are the same, so there is no price variation "
            "to explain."
        )
    else:
        print(f"R2 (Coefficient of Determination) = {r2:.6f}")
        if r2 < 0:
            print(
                f"  Total squared error is {-r2:.1%} higher than from always "
                "predicting the average price."
            )
        elif r2 == 0:
            print("  The model performs as well as always predicting the average price.")
        else:
            print(
                f"  Using mileage reduces total squared error by {r2:.1%} "
                "compared with always predicting the average price."
            )
    print(
        f"Evaluated on {len(prices)} rows from {DATA_PATH}: "
        "this measures fit on that dataset."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

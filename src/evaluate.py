import argparse
import sys
from pathlib import Path

from linear_regression import (
    LinearRegressionError,
    evaluate,
    load_data,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate MAE, RMSE and R2 for a trained model."
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "data.csv"),
        help="CSV file with km,price columns",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "data" / "model.json"),
        help="JSON file containing theta0 and theta1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mileages, prices = load_data(args.data)
        model = load_model(args.model)
        metrics = evaluate(mileages, prices, model)
    except (LinearRegressionError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"MAE  = {metrics.mae:.2f}")
    print(f"RMSE = {metrics.rmse:.2f}")
    if metrics.r2 is None:
        print("R2   = undefined (target prices are constant)")
    else:
        print(f"R2   = {metrics.r2:.6f}")
    print(
        f"Evaluated on {len(prices)} rows from {args.data}: "
        "this measures fit on that dataset."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

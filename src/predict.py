import argparse
import sys
from pathlib import Path

from linear_regression import LinearRegressionError, load_model, parse_mileage, predict_price

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a car price from its mileage.")
    parser.add_argument(
        "--model", default=str(PROJECT_ROOT / "data" / "model.json"), help="JSON file containing theta0 and theta1"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = load_model(args.model)
        mileage = parse_mileage(input("Mileage: "))
    except (LinearRegressionError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nNo mileage provided.", file=sys.stderr)
        return 1
    print(f"Estimated price: {predict_price(model, mileage):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

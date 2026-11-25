import math
import sys
from pathlib import Path

from linear_regression import load_model, predict_price

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"


def parse_mileage(text: str) -> float:
    """Validate mileage input and return it as a nonnegative finite float."""
    value = text.strip()
    if not value:
        raise ValueError("mileage cannot be empty")
    try:
        mileage = float(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid mileage {value!r}: expected a number"
        ) from exc

    if not math.isfinite(mileage):
        raise ValueError("mileage must be a finite number")
    if mileage < 0:
        raise ValueError("mileage cannot be negative")
    return mileage


def main() -> int:
    """Load the model, prompt for a mileage, and print the estimate."""
    try:
        model = load_model(str(MODEL_PATH))
        mileage = parse_mileage(input("Mileage: "))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nNo mileage provided.", file=sys.stderr)
        return 1
    print(f"Estimated price: {predict_price(model, mileage):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

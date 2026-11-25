import argparse
import json
import os
import sys
from pathlib import Path

from linear_regression import Model, load_data, predict_price, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"
LEARNING_RATE = 0.01


def save_model(model: Model, path: str) -> None:
    """Write theta0 and theta1 to a JSON file at path."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"theta0": model.theta0, "theta1": model.theta1},
            handle,
            indent=2,
        )
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train a linear regression model for car prices."
    )
    parser.add_argument(
        "--plot", action="store_true", help="display the data and fitted line"
    )
    parser.add_argument("--plot-output", metavar="PATH", help="save the graph to PATH")
    return parser


def plot_results(
    mileages: list[float],
    prices: list[float],
    model: Model,
    output: str | None,
    show: bool,
) -> None:
    """Plot the data and fitted line, optionally saving or displaying it."""
    try:
        import matplotlib

        if output and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib: pip install matplotlib"
        ) from exc

    figure, axes = plt.subplots()
    try:
        axes.scatter(mileages, prices, label="data")
        low = min(mileages)
        high = max(mileages)
        axes.plot(
            [low, high],
            [predict_price(model, low), predict_price(model, high)],
            color="tab:red",
            label="fit",
        )
        axes.set_xlabel("Mileage (km)")
        axes.set_ylabel("Price")
        axes.legend()
        axes.grid(True, alpha=0.3)
        if output:
            directory = os.path.dirname(os.path.abspath(output))
            os.makedirs(directory, exist_ok=True)
            figure.savefig(output)
            print(f"Plot saved to {output}")
        if show:
            plt.show()
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Train the model, save it, optionally plot, and report the result."""
    args = build_parser().parse_args(argv)
    try:
        mileages, prices = load_data(str(DATA_PATH))
        model, iterations, final_loss = train(
            mileages, prices, LEARNING_RATE
        )
        save_model(model, str(MODEL_PATH))
        if args.plot or args.plot_output:
            plot_results(mileages, prices, model, args.plot_output, args.plot)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Converged after {iterations} iterations")
    print(f"theta0 = {model.theta0:.6f}")
    print(f"theta1 = {model.theta1:.6f}")
    print(f"Final loss (MSE) = {final_loss:.6f}")
    print(f"Model saved to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

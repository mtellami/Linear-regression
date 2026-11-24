import argparse
import math
import os
import sys
from pathlib import Path

from linear_regression import DataError, TrainingError, load_data, predict_price, save_model, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a linear regression model for car prices.")
    parser.add_argument(
        "--data", default=str(PROJECT_ROOT / "data" / "data.csv"), help="CSV file with km,price columns"
    )
    parser.add_argument(
        "--model", default=str(PROJECT_ROOT / "data" / "model.json"), help="JSON file to save theta0 and theta1"
    )
    parser.add_argument("--learning-rate", type=positive_float, default=0.01)
    parser.add_argument("--max-iterations", type=positive_int, default=100000)
    parser.add_argument("--plot", action="store_true", help="display the data and fitted line")
    parser.add_argument("--plot-output", metavar="PATH", help="save the graph to PATH")
    return parser


def plot_results(mileages: list[float], prices: list[float], model, output: str | None, show: bool) -> None:
    try:
        import matplotlib

        if output and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib: pip install -r requirements-plot.txt") from exc

    figure, axes = plt.subplots()
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
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mileages, prices = load_data(args.data)
        result = train(mileages, prices, args.learning_rate, args.max_iterations)
        save_model(result.model, args.model)
        if args.plot or args.plot_output:
            plot_results(mileages, prices, result.model, args.plot_output, args.plot)
    except (DataError, TrainingError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.converged:
        print(f"Converged after {result.iterations} iterations")
    else:
        print(f"Warning: reached {args.max_iterations} iterations without converging", file=sys.stderr)
        print(f"Training stopped after {result.iterations} iterations")
    print(f"theta0 = {result.model.theta0:.6f}")
    print(f"theta1 = {result.model.theta1:.6f}")
    print(f"Final loss (MSE) = {result.final_loss:.6f}")
    print(f"Model saved to {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import csv
import json
import math
from dataclasses import dataclass

MAX_MILEAGE = 1_000_000
MAX_PRICE = 10_000_000


@dataclass(frozen=True)
class Model:
    theta0: float
    theta1: float


def predict_price(model: Model, mileage: float) -> float:
    """Estimate a price from mileage using theta0 + theta1 * mileage."""
    return model.theta0 + model.theta1 * mileage


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    """Return standardized values plus their mean and population std."""
    if len(values) < 2:
        raise ValueError("need at least 2 values to standardize")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        raise ValueError("mileage values have no variation")
    return [(value - mean) / std for value in values], mean, std


def mean_squared_error(
    values: list[float], prices: list[float], theta0: float, theta1: float
) -> float:
    """Return the mean squared error of the current predictions."""
    return sum(
        (theta0 + theta1 * value - price) ** 2 for value, price in zip(values, prices)
    ) / len(prices)


def gradient_step(
    values: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    learning_rate: float,
) -> tuple[float, float]:
    """Compute one simultaneous gradient step from the current coefficients."""
    errors = [theta0 + theta1 * value - price for value, price in zip(values, prices)]
    size = len(prices)
    slope0 = learning_rate * sum(errors) / size
    slope1 = (
        learning_rate
        * sum(error * value for error, value in zip(errors, values))
        / size
    )
    return theta0 - slope0, theta1 - slope1


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.01,
    tolerance: float = 1e-10,
) -> tuple[Model, int, float]:
    """Fit until loss improvement is below tolerance; return model, steps, loss."""
    if len(mileages) != len(prices) or len(mileages) < 2:
        raise ValueError("training needs at least 2 matching mileage and price values")

    values, mean, std = standardize(mileages)
    theta0 = 0.0
    theta1 = 0.0
    previous_loss = mean_squared_error(values, prices, theta0, theta1)
    iterations = 0

    while True:
        iterations += 1
        theta0, theta1 = gradient_step(values, prices, theta0, theta1, learning_rate)
        current_loss = mean_squared_error(values, prices, theta0, theta1)
        if not (
            math.isfinite(theta0)
            and math.isfinite(theta1)
            and math.isfinite(current_loss)
        ):
            raise ValueError(
                f"training produced non-finite values at iteration "
                f"{iterations}; lower the learning rate"
            )
        improvement = previous_loss - current_loss
        if 0.0 <= improvement < tolerance * max(previous_loss, 1.0):
            break
        previous_loss = current_loss

    theta1_original = theta1 / std
    theta0_original = theta0 - theta1_original * mean
    if not (math.isfinite(theta0_original) and math.isfinite(theta1_original)):
        raise ValueError("training produced non-finite parameters")
    return (
        Model(theta0_original, theta1_original),
        iterations,
        current_loss,
    )


######### UTILS #########
def load_data(path: str) -> tuple[list[float], list[float]]:
    """Read CSV data, rejecting values outside the supported car-data ranges."""
    mileages: list[float] = []
    prices: list[float] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mileage = float(row["km"])
                price = float(row["price"])
                if not 0 <= mileage <= MAX_MILEAGE:
                    raise ValueError(
                        f"line {reader.line_num}: mileage must be between "
                        f"0 and {MAX_MILEAGE:,} km"
                    )
                if not 0 <= price <= MAX_PRICE:
                    raise ValueError(
                        f"line {reader.line_num}: price must be between "
                        f"0 and {MAX_PRICE:,}"
                    )
                mileages.append(mileage)
                prices.append(price)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read data from {path}: {exc}") from exc
    if len(mileages) < 2:
        raise ValueError(f"{path}: need at least 2 valid rows")
    return mileages, prices


def load_model(path: str) -> Model:
    """Read model parameters from JSON, defaulting to zeros if absent."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return Model(0.0, 0.0)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read model file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"model file {path} must contain a JSON object")
    try:
        theta0 = float(payload["theta0"])
        theta1 = float(payload["theta1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"model file {path} must contain numeric theta0 and theta1"
        ) from exc

    if not (math.isfinite(theta0) and math.isfinite(theta1)):
        raise ValueError(f"model file {path} must contain finite parameters")
    return Model(theta0, theta1)

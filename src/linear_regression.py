import csv
import json
import math
from dataclasses import dataclass


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


def mse(z: list[float], prices: list[float], theta0: float, theta1: float) -> float:
    """Return the mean squared error of the current predictions."""
    return sum(
        (theta0 + theta1 * value - price) ** 2 for value, price in zip(z, prices)
    ) / len(prices)


def gradient_step(
    z: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    learning_rate: float,
) -> tuple[float, float]:
    """Compute one simultaneous gradient step from the current coefficients."""
    errors = [theta0 + theta1 * value - price for value, price in zip(z, prices)]
    size = len(prices)
    slope0 = learning_rate * sum(errors) / size
    slope1 = (
        learning_rate * sum(error * value for error, value in zip(errors, z)) / size
    )
    return theta0 - slope0, theta1 - slope1


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.01,
    max_iterations: int = 100000,
    tolerance: float = 1e-10,
) -> tuple[Model, int, float, bool]:
    """Fit the model and return parameters, iterations, loss, convergence."""
    if len(mileages) != len(prices) or len(mileages) < 2:
        raise ValueError("training needs at least 2 matching mileage and price values")

    z, mean, std = standardize(mileages)
    theta0 = 0.0
    theta1 = 0.0
    previous_loss = mse(z, prices, theta0, theta1)
    current_loss = previous_loss
    iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        theta0, theta1 = gradient_step(z, prices, theta0, theta1, learning_rate)
        current_loss = mse(z, prices, theta0, theta1)
        if not (
            math.isfinite(theta0)
            and math.isfinite(theta1)
            and math.isfinite(current_loss)
        ):
            raise ValueError(
                f"training produced non-finite values at iteration "
                f"{iteration}; lower the learning rate"
            )
        improvement = previous_loss - current_loss
        if 0.0 <= improvement < tolerance * max(previous_loss, 1.0):
            iterations = iteration
            converged = True
            break
        previous_loss = current_loss
    else:
        iterations = max_iterations

    theta1_original = theta1 / std
    theta0_original = theta0 - theta1_original * mean
    if not (math.isfinite(theta0_original) and math.isfinite(theta1_original)):
        raise ValueError("training produced non-finite parameters")
    return (
        Model(theta0_original, theta1_original),
        iterations,
        current_loss,
        converged,
    )


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


def load_data(path: str) -> tuple[list[float], list[float]]:
    """Read mileage and price columns from a CSV file."""
    mileages: list[float] = []
    prices: list[float] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                mileages.append(float(row["km"]))
                prices.append(float(row["price"]))
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

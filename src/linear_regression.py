import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional


class LinearRegressionError(Exception):
    pass


class DataError(LinearRegressionError):
    pass


class ModelError(LinearRegressionError):
    pass


class TrainingError(LinearRegressionError):
    pass


@dataclass(frozen=True)
class Model:
    theta0: float
    theta1: float


@dataclass(frozen=True)
class TrainingResult:
    model: Model
    iterations: int
    final_loss: float
    converged: bool


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: Optional[float]


def parse_mileage(text: str) -> float:
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


def _parse_csv_number(
    text: str, path: str, row_number: int, column: str
) -> float:
    value = text.strip()
    if not value:
        raise DataError(
            f"{path}: row {row_number} has an empty {column} value"
        )
    try:
        number = float(value)
    except ValueError as exc:
        raise DataError(
            f"{path}: row {row_number} has a non-numeric {column} "
            f"value {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise DataError(
            f"{path}: row {row_number} has a non-finite {column} "
            f"value {value!r}"
        )
    if number < 0:
        raise DataError(
            f"{path}: row {row_number} has a negative {column} value {value!r}"
        )
    return number


def load_data(path: str) -> tuple[list[float], list[float]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DataError(
                    f"{path}: missing header row with km and price columns"
                ) from exc
            columns = [name.strip() for name in header]
            if "km" not in columns or "price" not in columns:
                raise DataError(
                    f"{path}: expected km and price columns, got {header!r}"
                )
            km_index = columns.index("km")
            price_index = columns.index("price")
            mileages: list[float] = []
            prices: list[float] = []
            for row_number, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) <= max(km_index, price_index):
                    raise DataError(
                        f"{path}: row {row_number} is missing values"
                    )
                mileages.append(
                    _parse_csv_number(row[km_index], path, row_number, "km")
                )
                prices.append(
                    _parse_csv_number(
                        row[price_index], path, row_number, "price"
                    )
                )
    except OSError as exc:
        raise DataError(f"cannot read data file {path}: {exc}") from exc
    if len(mileages) < 2:
        raise DataError(
            f"{path}: need at least 2 valid rows, found {len(mileages)}"
        )
    return mileages, prices


def predict_price(model: Model, mileage: float) -> float:
    return model.theta0 + model.theta1 * mileage


def save_model(model: Model, path: str) -> None:
    payload = {"theta0": model.theta0, "theta1": model.theta1}
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".model-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def load_model(path: str) -> Model:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return Model(0.0, 0.0)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read model file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError(f"model file {path} must contain a JSON object")
    for key in ("theta0", "theta1"):
        if key not in payload:
            raise ModelError(f"model file {path} is missing {key!r}")
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ModelError(
                f"model file {path} has an invalid {key!r}: {value!r}"
            )
    return Model(float(payload["theta0"]), float(payload["theta1"]))


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    if len(values) < 2:
        raise TrainingError("need at least 2 values to standardize")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        raise TrainingError("mileage values have no variation")
    return [(value - mean) / std for value in values], mean, std


def mse(
    z: list[float], prices: list[float], theta0: float, theta1: float
) -> float:
    return sum(
        (theta0 + theta1 * value - price) ** 2
        for value, price in zip(z, prices)
    ) / len(prices)


def gradient_step(
    z: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
    learning_rate: float,
) -> tuple[float, float]:
    errors = [
        theta0 + theta1 * value - price for value, price in zip(z, prices)
    ]
    size = len(prices)
    slope0 = learning_rate * sum(errors) / size
    slope1 = (
        learning_rate
        * sum(error * value for error, value in zip(errors, z))
        / size
    )
    return theta0 - slope0, theta1 - slope1


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.01,
    max_iterations: int = 100000,
    tolerance: float = 1e-10,
) -> TrainingResult:
    if len(mileages) != len(prices) or len(mileages) < 2:
        raise TrainingError(
            "training needs at least 2 matching mileage and price values"
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise TrainingError("learning rate must be a positive finite number")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise TrainingError("max iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise TrainingError("tolerance must be a nonnegative finite number")

    z, mean, std = standardize(mileages)
    theta0 = 0.0
    theta1 = 0.0
    previous_loss = mse(z, prices, theta0, theta1)
    current_loss = previous_loss
    iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        theta0, theta1 = gradient_step(
            z, prices, theta0, theta1, learning_rate
        )
        if not math.isfinite(theta0) or not math.isfinite(theta1):
            raise TrainingError(
                "training produced non-finite coefficients at iteration "
                f"{iteration}; lower the learning rate"
            )
        current_loss = mse(z, prices, theta0, theta1)
        if not math.isfinite(current_loss):
            raise TrainingError(
                "training produced a non-finite loss at iteration "
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
    if not math.isfinite(theta0_original) or not math.isfinite(
        theta1_original
    ):
        raise TrainingError("training produced non-finite parameters")
    return TrainingResult(
        Model(theta0_original, theta1_original),
        iterations,
        current_loss,
        converged,
    )


def evaluate(
    mileages: list[float], prices: list[float], model: Model
) -> Metrics:
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
    r2: Optional[float] = (
        None if total_variance == 0.0 else 1.0 - squared_error / total_variance
    )
    return Metrics(
        mae=sum(abs(error) for error in errors) / size,
        rmse=math.sqrt(squared_error / size),
        r2=r2,
    )

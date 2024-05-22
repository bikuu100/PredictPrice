# app.py

from exception import InputError, CalculationError
from loggers import setup_logging

logger = setup_logging()

def divide(a, b):
    try:
        if b == 0:
            raise InputError("Division by zero is not allowed")
        return a / b
    except InputError as e:
        logger.error(f"Input error: {e.message}")
        return None

def perform_calculation(a, b):
    try:
        result = divide(a, b)
        if result is None:
            raise CalculationError("Failed to perform division")
        return result
    except CalculationError as e:
        logger.error(f"Calculation error: {e.message}")
        return None

if __name__ == "__main__":
    result = perform_calculation(10, 0)
    if result is not None:
        logger.info(f"Calculation result: {result}")
    else:
        logger.warning("Calculation could not be completed")

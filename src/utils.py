"""
Shared utility helpers used across the application:
- Indian Rupee currency formatting
- Safe numeric coercion
- Small generic helpers
"""
from __future__ import annotations

import numpy as np


def format_inr(amount: float, decimals: int = 0) -> str:
    """
    Format a number using the Indian numbering system (lakh / crore grouping)
    with a Rupee symbol, e.g. 1234567 -> '₹12,34,567'.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "₹0"

    if np.isnan(amount) or np.isinf(amount):
        return "₹0"

    negative = amount < 0
    amount = abs(amount)

    if decimals > 0:
        whole = int(amount)
        frac = round(amount - whole, decimals)
        frac_str = f"{frac:.{decimals}f}".split(".")[1]
    else:
        whole = int(round(amount))
        frac_str = None

    s = str(whole)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ",".join(parts) + "," + last3
    else:
        formatted = s

    result = f"₹{formatted}"
    if frac_str is not None:
        result += f".{frac_str}"
    if negative:
        result = f"-{result}"
    return result


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, returning a default on failure."""
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide while guarding against division by zero / invalid input."""
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value between low and high."""
    return max(low, min(high, value))


def pct(value: float, decimals: int = 1) -> str:
    """Format a fraction (0-1) or plain percentage number as a percentage string."""
    value = safe_float(value)
    return f"{round(value, decimals)}%"


REQUIRED_CSV_COLUMNS = ["Date", "Category", "Amount", "Type"]
VALID_TYPES = {"Income", "Expense"}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

"""Column names shared by the fixtures and the tests.

Mirrors the upstream churn dataset's shape without copying its rows: three numeric fields, three categorical
fields, and a binary target.
"""

from __future__ import annotations

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]
TARGET = "Churn"
FEATURE_COLUMNS = NUMERIC + CATEGORICAL

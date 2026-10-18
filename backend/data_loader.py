
from pathlib import Path

import pandas as pd


# Find the project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the path to our product dataset
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"


def load_products() -> pd.DataFrame:
    """Load product data from the CSV file."""
    products = pd.read_csv(DATA_PATH)
    return products


def validate_products(products: pd.DataFrame) -> None:
    """Check basic product dataset quality."""

    required_columns = [
        "product_id",
        "name",
        "category",
        "brand",
        "price",
        "rating",
        "ram_gb",
        "storage_gb",
        "description",
    ]

    # Check required columns
    assert all(
        column in products.columns for column in required_columns
    ), "Missing required columns"

    # Check that product IDs are unique
    assert products["product_id"].is_unique, "Duplicate product IDs"

    # Check required text fields
    assert products["name"].notna().all(), "Missing product names"
    assert products["category"].notna().all(), "Missing categories"

    # Check prices
    assert products["price"].notna().all(), "Missing prices"
    assert (products["price"] > 0).all(), "Prices must be positive"

    print("All basic product data checks passed!")



if __name__ == "__main__":
    products = load_products()
    validate_products(products)

    print("\nDataset shape:")
    print(products.shape)

    print("\nMissing values:")
    print(products.isnull().sum())

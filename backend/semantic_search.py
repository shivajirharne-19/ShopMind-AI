
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"

# Load the product catalog
products = pd.read_csv(DATA_PATH)

# Combine useful product information into searchable text
products["search_text"] = (
    products["name"].fillna("")
    + " "
    + products["category"].fillna("")
    + " "
    + products["brand"].fillna("")
    + " "
    + products["description"].fillna("")
)

# Load a pretrained embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert product text into embeddings
product_embeddings = model.encode(
    products["search_text"].tolist(),
    normalize_embeddings=True,
)

def search_products(
    query: str,
    top_k: int = 5,
    max_price: float | None = None,
    category: str | None = None,
    min_ram_gb: float | None = None,
    min_storage_gb: float | None = None,
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = product_embeddings @ query_embedding

    results = products.copy()
    results["similarity"] = scores

    if max_price is not None:
        results = results[results["price"] <= max_price]

    if category is not None:
        results = results[
            results["category"].str.lower() == category.strip().lower()
        ]

    if min_ram_gb is not None:
        results = results[
            results["ram_gb"].notna()
            & (results["ram_gb"] >= min_ram_gb)
        ]

    if min_storage_gb is not None:
        results = results[
            results["storage_gb"].notna()
            & (results["storage_gb"] >= min_storage_gb)
        ]

    results = results.nlargest(top_k, "similarity")

    return results[
        [
            "product_id",
            "name",
            "category",
            "brand",
            "price",
            "rating",
            "ram_gb",
            "storage_gb",
            "description",
            "similarity",
        ]
    ]


if __name__ == "__main__":
    query = "I need a portable computer for college"

    results = search_products(
        query=query,
        top_k=5,
        max_price=70000,
        category="Laptop",
        min_ram_gb=8,
    )

    print(f"Search query: {query}\n")
    print(results.to_string(index=False))
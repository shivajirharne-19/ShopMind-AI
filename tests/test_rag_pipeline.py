
from unittest.mock import patch

import pandas as pd

from backend.rag_pipeline import ask_shopmind


def test_no_matching_products():
    """No-match requests should return a clear answer without calling Groq."""

    with patch(
        "backend.rag_pipeline.search_products",
        return_value=pd.DataFrame(),
    ), patch(
        "backend.rag_pipeline.client.chat.completions.create"
    ) as mock_llm:
        result = ask_shopmind(
            query="Find a laptop",
            max_price=1000,
            category="Laptop",
        )

    assert "No matching products" in result["answer"]
    assert result["products"].empty
    mock_llm.assert_not_called()


def test_product_filters_are_respected():
    """Every returned product must satisfy the requested filters."""

    products = pd.DataFrame(
        [
            {
                "name": "Laptop A",
                "price": 60000,
                "ram_gb": 16,
                "storage_gb": 512,
                "similarity": 0.9,
            },
            {
                "name": "Laptop B",
                "price": 50000,
                "ram_gb": 8,
                "storage_gb": 512,
                "similarity": 0.8,
            },
        ]
    )

    with patch(
        "backend.rag_pipeline.search_products",
        return_value=products,
    ), patch(
        "backend.rag_pipeline.client.chat.completions.create"
    ) as mock_llm:
        mock_llm.return_value.choices[0].message.content = (
            "Laptop A and Laptop B match the supplied records."
        )

        result = ask_shopmind(
            query="Recommend a laptop",
            max_price=70000,
            category="Laptop",
            min_ram_gb=8,
            min_storage_gb=512,
        )

    returned = result["products"]

    assert not returned.empty
    assert (returned["price"] <= 70000).all()
    assert (returned["ram_gb"] >= 8).all()
    assert (returned["storage_gb"] >= 512).all()


def test_response_structure():
    """The function should return an answer and a product DataFrame."""

    products = pd.DataFrame(
        [
            {
                "name": "Laptop A",
                "category": "Laptop",
                "brand": "Example",
                "price": 60000,
                "rating": 4.2,
                "ram_gb": 16,
                "storage_gb": 512,
                "description": "Laptop for study and productivity",
                "product_id": "TEST001",
                "similarity": 0.9,
            }
        ]
    )

    with patch(
        "backend.rag_pipeline.search_products",
        return_value=products,
    ), patch(
        "backend.rag_pipeline.client.chat.completions.create"
    ) as mock_llm:
        mock_llm.return_value.choices[0].message.content = (
            "Laptop A is listed for ₹60,000."
        )

        result = ask_shopmind(query="Recommend a laptop")

    assert isinstance(result, dict)
    assert "answer" in result
    assert "products" in result
    assert isinstance(result["answer"], str)
    assert result["answer"]
    assert isinstance(result["products"], pd.DataFrame)


from backend.semantic_search import search_products


def test_budget_filter():
    """Every returned product must be within the budget."""

    results = search_products(
        query="laptop for college",
        top_k=20,
        max_price=70000,
        category="Laptop",
    )

    assert not results.empty
    assert (results["price"] <= 70000).all()


def test_category_filter_is_case_insensitive():
    """Category matching should ignore letter case and extra spaces."""

    results = search_products(
        query="laptop",
        top_k=20,
        category="  LAPTOP  ",
    )

    assert not results.empty
    assert results["category"].str.lower().eq("laptop").all()


def test_ram_and_storage_filters():
    """Returned products must meet both minimum specifications."""

    results = search_products(
        query="laptop for college",
        top_k=20,
        category="Laptop",
        min_ram_gb=8,
        min_storage_gb=512,
    )

    assert not results.empty
    assert (results["ram_gb"] >= 8).all()
    assert (results["storage_gb"] >= 512).all()


def test_no_products_match_budget():
    """An impossible budget should return an empty DataFrame."""

    results = search_products(
        query="laptop",
        top_k=20,
        max_price=1000,
        category="Laptop",
    )

    assert results.empty

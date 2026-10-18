
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from backend.semantic_search import search_products


# --------------------------------------------------
# 1. Load environment variables and initialize Groq
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your project-root .env file."
    )

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-120b"


# --------------------------------------------------
# 2. Build context from retrieved product records
# --------------------------------------------------

def build_product_context(products):
    """Convert retrieved product rows into readable context."""

    if products.empty:
        return "No products matched the search and filter criteria."

    lines = []

    fields = [
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

    for _, product in products.iterrows():
        details = []

        for field in fields:
            if field not in products.columns:
                continue

            value = product[field]

            # Represent missing information honestly
            if value is None or str(value).lower() in {
                "nan",
                "none",
                "<na>",
            }:
                value = "Unknown"

            # Format prices in Indian rupees
            if field == "price" and value != "Unknown":
                value = f"₹{float(value):,.0f}"

            details.append(f"{field}: {value}")

        lines.append("\n".join(details))

    return "\n\n---\n\n".join(lines)


# --------------------------------------------------
# 3. Retrieve products and generate an answer
# --------------------------------------------------

def ask_shopmind(
    query: str,
    top_k: int = 5,
    max_price: float | None = None,
    category: str | None = None,
    min_ram_gb: float | None = None,
    min_storage_gb: float | None = None,
):
    """Retrieve matching products and generate a grounded answer."""

    # Retrieve products using semantic search and filters
    products = search_products(
        query=query,
        top_k=top_k,
        max_price=max_price,
        category=category,
        min_ram_gb=min_ram_gb,
        min_storage_gb=min_storage_gb,
    )

    # Return a deterministic response if no products match.
    # This avoids an unnecessary LLM request.
    if products.empty:
        return {
            "answer": (
                "No matching products were found for your search "
                "and selected filters. Try adjusting your budget "
                "or product requirements."
            ),
            "products": products,
        }

    # Convert retrieved products into context for the LLM
    context = build_product_context(products)

    # System instructions: keep recommendations grounded in data
    system_prompt = """
You are ShopMind AI, a shopping assistant for products in India.

Follow these rules strictly:

1. Use only the supplied product context for product facts.
2. Never invent prices, ratings, specifications, features,
   availability, or performance claims.
3. Format all prices in Indian rupees (₹).
4. Never claim a product is the best, fastest, lightest,
   or most powerful unless the context explicitly supports it.
5. Treat missing specifications as unknown. Never guess them.
6. Recommend only products included in the retrieved context.
7. Respect the user's requirements and the applied filters.
8. Explain recommendations using facts from the context.
9. You may make a cautious inference from a description,
   but label it as an inference rather than a verified fact.
10. Do not claim a product meets a specific software requirement
    unless the context provides supporting evidence.
11. The dataset contains sample product information.
    Do not claim prices or ratings are verified current market data.
12. If the context does not support a claim, say so honestly.
13. Do not invent product weights, battery life, processors,
    benchmarks, or other missing specifications.
14. Keep the answer concise and useful.
15. Prefer a short list or table, and explain recommendations
    using the available product facts.
"""

    # Prepare the user's question and applied filters
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""
User question:
{query}

Applied filters:
Maximum price: {
    f"₹{max_price:,.0f}" if max_price is not None else "No limit"
}
Category: {category if category is not None else "Any"}
Minimum RAM: {
    f"{min_ram_gb:g} GB" if min_ram_gb is not None else "No minimum"
}
Minimum storage: {
    f"{min_storage_gb:g} GB"
    if min_storage_gb is not None
    else "No minimum"
}

Retrieved product context:
{context}

Recommend suitable products using only the supplied product records.
Do not include products outside the retrieved context.
""",
        },
    ]

    # Generate the recommendation using Groq
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    answer = response.choices[0].message.content

    # Handle an unexpectedly empty model response
    if not answer:
        answer = (
            "I couldn't generate a recommendation this time. "
            "Please try your question again."
        )

    return {
        "answer": answer,
        "products": products,
    }


# --------------------------------------------------
# 4. Run a manual test when executing this file
# --------------------------------------------------

if __name__ == "__main__":
    query = "Recommend a portable laptop for college"

    result = ask_shopmind(
        query=query,
        top_k=5,
        max_price=70000,
        category="Laptop",
        min_ram_gb=8,
        min_storage_gb=512,
    )

    print("\nSHOPMIND AI ANSWER\n")
    print(result["answer"])

    print("\nRETRIEVED PRODUCTS\n")

    if result["products"].empty:
        print("No matching products found.")
    else:
        columns_to_show = [
            "name",
            "price",
            "ram_gb",
            "storage_gb",
            "similarity",
        ]

        print(
            result["products"][columns_to_show].to_string(index=False)
        )

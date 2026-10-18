
from backend.semantic_search import search_products


# --------------------------------------------------
# 1. Define evaluation cases
# --------------------------------------------------

EVALUATION_CASES = [
    {
        "name": "College laptop",
        "query": "lightweight laptop for study",
        "category": "Laptop",
        "expected_relevant": {
            "Lenovo IdeaPad Slim 5",
        },
    },
    {
        "name": "Photography smartphone",
        "query": "smartphone focused on photography",
        "category": "Smartphone",
        "expected_relevant": {
            "Samsung Galaxy S24",
            "Google Pixel 8a",
        },
    },
     {
        "name": "Noise-cancelling headphones",
        "query": "wireless headphones with noise cancellation",
        "category": "Headphones",
        "expected_relevant": {
            "Sony WH-1000XM5",
            "JBL Tune 770NC",
        },
    },
    {
        "name": "Fitness smartwatch",
        "query": "smartwatch for fitness tracking",
        "category": "Smartwatch",
        "expected_relevant": {
            "Apple Watch SE",
            "Samsung Galaxy Watch 6",
        },
    },
    {
        "name": "Tablet for reading",
        "query": "tablet for reading and entertainment",
        "category": "Tablet",
        "expected_relevant": {
            "Lenovo Tab P12",
        },
    },
]

# --------------------------------------------------
# 2. Evaluate retrieval quality and filters
# --------------------------------------------------

def evaluate_retrieval():
    total_relevant = 0
    total_retrieved_relevant = 0
    total_queries = len(EVALUATION_CASES)
    filter_correct_queries = 0

    print("\nSHOPMIND AI RETRIEVAL EVALUATION")
    print("=" * 55)

    for case in EVALUATION_CASES:
        results = search_products(
            query=case["query"],
            top_k=5,
            category=case["category"],
        )

        expected = case["expected_relevant"]
        retrieved_names = set(results["name"].tolist())

        relevant_found = expected.intersection(retrieved_names)

        total_relevant += len(expected)
        total_retrieved_relevant += len(relevant_found)

        # Verify that every result matches the requested category
        category_correct = (
            results.empty
            or results["category"]
            .str.lower()
            .eq(case["category"].lower())
            .all()
        )

        if category_correct:
            filter_correct_queries += 1

        print(f"\nEvaluation: {case['name']}")
        print(f"Query: {case['query']}")
        print(f"Expected relevant products: {sorted(expected)}")
        print(f"Retrieved products: {results['name'].tolist()}")
        print(f"Relevant products found: {sorted(relevant_found)}")
        print(f"Category filter correct: {category_correct}")

        if not relevant_found:
            print("Result: No expected relevant product found.")
        else:
            print("Result: At least one expected relevant product found.")

    # Calculate aggregate metrics
    recall_at_5 = (
        total_retrieved_relevant / total_relevant
        if total_relevant
        else 0
    )

    filter_accuracy = (
        filter_correct_queries / total_queries
        if total_queries
        else 0
    )

    print("\n" + "=" * 55)
    print("FINAL EVALUATION METRICS")
    print("=" * 55)
    print(f"Queries evaluated: {total_queries}")
    print(f"Recall@5: {recall_at_5:.2%}")
    print(f"Category filter accuracy: {filter_accuracy:.2%}")


# --------------------------------------------------
# 3. Run evaluation
# --------------------------------------------------

if __name__ == "__main__":
    evaluate_retrieval()

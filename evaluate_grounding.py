
import re
import pandas as pd


# --------------------------------------------------
# 1. Load the product dataset
# --------------------------------------------------

products = pd.read_csv("data/products.csv")

# Normalize product names for reliable matching.
products["name"] = products["name"].astype(str).str.strip()


# --------------------------------------------------
# 2. Check whether an answer is grounded in the dataset
# --------------------------------------------------

def check_grounding(product_name: str, answer: str) -> dict:
    """
    Check selected factual claims against a product's dataset record.

    This is a basic rule-based checker, not a complete
    semantic evaluation of every statement in the answer.
    """

    matches = products[
        products["name"].str.lower() == product_name.strip().lower()
    ]

    if matches.empty:
        return {
            "product_found": False,
            "passed": False,
            "issues": ["Product was not found in the dataset."],
        }

    product = matches.iloc[0]
    answer_lower = answer.lower()
    issues = []

    # Build context from the selected product's available fields.
    context_parts = []

    for field in [
        "name",
        "category",
        "brand",
        "description",
        "price",
        "rating",
        "ram_gb",
        "storage_gb",
        "weight",
        "weight_kg",
        "weight_g",
        "weight_grams",
        "battery",
        "battery_life",
        "processor",
        "cpu",
        "gpu",
        "graphics",
        "benchmark",
    ]:
        if field in product.index and pd.notna(product[field]):
            context_parts.append(str(product[field]))

    context = " ".join(context_parts).lower()

    # --------------------------------------------------
    # Check 1: Specific weight claims
    # --------------------------------------------------

    weight_claim = re.search(
        r"\bweighs?\s+\d+(?:\.\d+)?\s*"
        r"(?:kg|kilograms?|g|grams?)\b",
        answer_lower,
    )

    # IMPORTANT:
    # Do not search for "weight" in the description.
    # For example, "lightweight" does not establish a
    # specific product weight.
    weight_fields = [
        "weight",
        "weight_kg",
        "weight_g",
        "weight_grams",
    ]

    has_weight_data = any(
        field in product.index and pd.notna(product[field])
        for field in weight_fields
    )

    if weight_claim and not has_weight_data:
        issues.append(
            "Specific product weight is not in the dataset."
        )

    # --------------------------------------------------
    # Check 2: Battery-life claims
    # --------------------------------------------------

    battery_claim = (
        "battery life" in answer_lower
        or re.search(
            r"\bbattery\b.{0,40}\b\d+\s*hours?\b",
            answer_lower,
        )
    )

    battery_fields = [
        "battery",
        "battery_life",
        "battery_hours",
    ]

    has_battery_data = any(
        field in product.index and pd.notna(product[field])
        for field in battery_fields
    )

    if battery_claim and not has_battery_data:
        issues.append(
            "Battery information is not in the dataset."
        )

    # --------------------------------------------------
    # Check 3: Processor, graphics, and benchmark claims
    # --------------------------------------------------

    technical_claims = {
        "processor": ["processor", "cpu"],
        "graphics": [
            "gpu",
            "graphics card",
            "graphics processor",
        ],
        "benchmark": ["benchmark", "benchmark score"],
    }

    technical_fields = {
        "processor": ["processor", "cpu"],
        "graphics": ["gpu", "graphics"],
        "benchmark": ["benchmark"],
    }

    for claim_type, keywords in technical_claims.items():
        if any(keyword in answer_lower for keyword in keywords):
            fields = technical_fields[claim_type]

            has_technical_data = any(
                field in product.index and pd.notna(product[field])
                for field in fields
            )

            # Also check the available product text for the
            # specific type of technical information.
            has_technical_data = has_technical_data or any(
                keyword in context for keyword in keywords
            )

            if not has_technical_data:
                issues.append(
                    f"{claim_type.title()} information is not "
                    "supported by the dataset."
                )

    # --------------------------------------------------
    # Check 4: Exact rupee amounts
    # --------------------------------------------------

    mentioned_prices = re.findall(
        r"₹\s*([\d,]+(?:\.\d{1,2})?)",
        answer,
    )

    if mentioned_prices and "price" in product.index:
        actual_price = pd.to_numeric(
            product["price"], errors="coerce"
        )

        if pd.notna(actual_price):
            for price_text in mentioned_prices:
                mentioned_price = float(
                    price_text.replace(",", "")
                )

                if abs(mentioned_price - float(actual_price)) > 0.01:
                    issues.append(
                        f"Answer mentions ₹{mentioned_price:,.2f}, "
                        f"but the dataset lists "
                        f"₹{float(actual_price):,.2f} "
                        f"for {product_name}."
                    )

    return {
        "product_found": True,
        "passed": len(issues) == 0,
        "issues": issues,
    }


# --------------------------------------------------
# 3. Define test cases
# --------------------------------------------------

TEST_CASES = [
    {
        "name": "Supported description",
        "product": "Lenovo IdeaPad Slim 5",
        "answer": (
            "The Lenovo IdeaPad Slim 5 is described as a "
            "lightweight laptop for study and productivity."
        ),
        "expected_pass": True,
    },
    {
        "name": "Unsupported weight",
        "product": "Lenovo IdeaPad Slim 5",
        "answer": (
            "The Lenovo IdeaPad Slim 5 weighs 1.2 kg."
        ),
        "expected_pass": False,
    },
    {
        "name": "Unsupported battery life",
        "product": "Lenovo IdeaPad Slim 5",
        "answer": (
            "The Lenovo IdeaPad Slim 5 has a battery life "
            "of 12 hours."
        ),
        "expected_pass": False,
    },
    {
        "name": "Incorrect price",
        "product": "Lenovo IdeaPad Slim 5",
        "answer": (
            "The Lenovo IdeaPad Slim 5 costs ₹50,000."
        ),
        "expected_pass": False,
    },
]


# --------------------------------------------------
# 4. Run the evaluation
# --------------------------------------------------

def evaluate_grounding():
    passed_cases = 0

    print("\nSHOPMIND AI GROUNDING EVALUATION")
    print("=" * 55)

    for case in TEST_CASES:
        result = check_grounding(
            product_name=case["product"],
            answer=case["answer"],
        )

        test_passed = (
            result["passed"] == case["expected_pass"]
        )

        if test_passed:
            passed_cases += 1

        print(f"\nTest: {case['name']}")
        print(f"Answer: {case['answer']}")
        print(f"Checker result: {result['passed']}")
        print(
            f"Test result: {'PASS' if test_passed else 'FAIL'}"
        )

        for issue in result["issues"]:
            print(f"  Issue: {issue}")

    total_cases = len(TEST_CASES)
    pass_rate = passed_cases / total_cases if total_cases else 0

    print("\n" + "=" * 55)
    print(f"Tests passed: {passed_cases}/{total_cases}")
    print(f"Test pass rate: {pass_rate:.2%}")


if __name__ == "__main__":
    evaluate_grounding()

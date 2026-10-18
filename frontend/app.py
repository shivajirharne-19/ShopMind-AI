
import requests
import pandas as pd
import streamlit as st

# FastAPI backend address
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ ShopMind AI")
st.subheader("Your AI-powered shopping assistant")

st.write(
    "Describe what you need, set your preferences, "
    "and get recommendations from the product dataset."
)

# User query
query = st.text_input(
    "What are you looking for?",
    placeholder="Example: A portable laptop for college",
)

# Product filters
st.sidebar.header("Product Preferences")

category = st.sidebar.selectbox(
    "Category",
    ["Any", "Laptop", "Tablet", "Smartphone", "Headphones"],
)

budget_option = st.sidebar.selectbox(
    "Maximum budget",
    [
        "No limit",
        "₹30,000",
        "₹50,000",
        "₹70,000",
        "₹1,00,000",
    ],
)

ram_option = st.sidebar.selectbox(
    "Minimum RAM",
    ["No minimum", "4 GB", "8 GB", "16 GB", "32 GB"],
)

storage_option = st.sidebar.selectbox(
    "Minimum storage",
    ["No minimum", "128 GB", "256 GB", "512 GB", "1024 GB"],
)

top_k = st.sidebar.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5,
)

# Convert the user's selections into API parameters
budget_map = {
    "No limit": None,
    "₹30,000": 30000,
    "₹50,000": 50000,
    "₹70,000": 70000,
    "₹1,00,000": 100000,
}

ram_map = {
    "No minimum": None,
    "4 GB": 4,
    "8 GB": 8,
    "16 GB": 16,
    "32 GB": 32,
}

storage_map = {
    "No minimum": None,
    "128 GB": 128,
    "256 GB": 256,
    "512 GB": 512,
    "1024 GB": 1024,
}

if st.button("Find Products", type="primary"):
    if not query.strip():
        st.warning("Please enter a shopping question.")

    else:
        payload = {
            "query": query.strip(),
            "top_k": top_k,
            "max_price": budget_map[budget_option],
            "category": (
                None if category == "Any" else category
            ),
            "min_ram_gb": ram_map[ram_option],
            "min_storage_gb": storage_map[storage_option],
        }

        try:
            with st.spinner(
                "Searching products and generating recommendations..."
            ):
                response = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=120,
                )

                response.raise_for_status()
                result = response.json()

            st.session_state["shopmind_result"] = result

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI backend. "
                "Make sure the API server is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request took too long. Please try again."
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"API request failed: {exc}")

        except ValueError:
            st.error("The backend returned an invalid response.")

# Display the latest successful result
result = st.session_state.get("shopmind_result")

if result:
    st.divider()
    st.header("AI Recommendation")
    st.markdown(result["answer"])

    st.divider()
    st.header("Matching Products")
    st.write(f"Products found: {result['count']}")

    if result["products"]:
        products_df = pd.DataFrame(result["products"])

        columns = [
            "name",
            "brand",
            "category",
            "price",
            "rating",
            "ram_gb",
            "storage_gb",
            "similarity",
        ]

        available_columns = [
            column
            for column in columns
            if column in products_df.columns
        ]

        st.dataframe(
            products_df[available_columns],
            width="stretch",
            hide_index=True,
)

        st.caption(
            "Prices and specifications come from the project's "
            "sample dataset and are not verified live market data."
        )

    else:
        st.info(
            "No products matched your selected requirements. "
            "Try adjusting your budget or specifications."
        )

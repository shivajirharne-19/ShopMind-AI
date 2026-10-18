# ShopMind AI 🛍️

An AI-powered shopping assistant that combines semantic product search, metadata filtering, and Retrieval-Augmented Generation (RAG) to provide product recommendations grounded in a product dataset.

## Overview

ShopMind AI helps users find products by describing their needs in natural language. It retrieves relevant products, applies user-selected filters, and generates recommendations using the retrieved product information.

This project demonstrates practical skills in Generative AI, semantic search, RAG pipelines, API development, and interactive application development.

## Features

- **Semantic search:** Find products using natural-language queries.
- **Product filtering:** Filter by category, maximum price, minimum RAM, and minimum storage.
- **RAG-based recommendations:** Generate recommendations using retrieved product records.
- **FastAPI backend:** Expose product recommendations through an API.
- **Streamlit frontend:** Interact with the assistant through a web interface.
- **Automated tests:** Validate filtering, response structure, and no-match handling.
- **Evaluation scripts:** Evaluate retrieval performance and check selected grounding cases.

## Tech Stack

- Python
- Sentence Transformers
- Groq API
- Retrieval-Augmented Generation (RAG)
- FastAPI
- Streamlit
- Pandas
- Pydantic
- Pytest

## Architecture

1. The user enters a shopping query and optional product filters in Streamlit.
2. The frontend sends the request to the FastAPI backend.
3. The semantic search component retrieves matching products from the dataset.
4. The RAG pipeline provides the retrieved product information to the language model.
5. The API returns the recommendation and matching product records.
6. Streamlit displays the recommendation and product table.

## Project Structure

```text
ShopMind-AI/
├── backend/
│   ├── api.py
│   ├── data_loader.py
│   ├── groq_test.py
│   ├── rag_pipeline.py
│   └── semantic_search.py
├── data/
│   └── products.csv
├── frontend/
│   └── app.py
├── tests/
│   └── test_rag_pipeline.py
├── evaluate_grounding.py
├── evaluate_retrieval.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ShopMind-AI
```

### 2. Create and activate a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your `.env` file or expose your API key.

## Run the Application

Open two terminals in the project root and activate the virtual environment in each.

### Terminal 1 — FastAPI backend

```cmd
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

### Terminal 2 — Streamlit frontend

```cmd
python -m streamlit run frontend/app.py
```

Open the local URL displayed by Streamlit, usually http://localhost:8501.

## Testing

Run the automated tests:

```cmd
python -m pytest tests/test_rag_pipeline.py -v
```

Latest recorded result: **7 tests passed**.

Run the retrieval evaluation:

```cmd
python evaluate_retrieval.py
```

Run the grounding checks:

```cmd
python evaluate_grounding.py
```

The recorded retrieval evaluation achieved 100% Recall@5 across five manually labeled queries, with 100% category-filter accuracy on the tested cases. The grounding checker passed four selected test cases.

These results describe small evaluation sets and should not be interpreted as a guarantee of overall system accuracy or hallucination-free responses.

## Dataset

The project uses a sample product dataset containing product names, categories, brands, prices, ratings, RAM, storage, and descriptions.

Prices and specifications are sample data and are not verified live market information.

## Limitations

- Recommendations are limited to the available dataset.
- The application does not verify real-time prices, inventory, or product availability.
- Retrieval and grounding evaluations cover a small set of manually selected examples.
- Recommendation quality depends on the retrieved records and language-model output.

## Future Improvements

- Expand the product dataset.
- Add more automated API and frontend tests.
- Improve grounding and hallucination evaluation.
- Add product comparison and recommendation explanations.
- Deploy the application when production configuration and security are ready.

## Author

Developed as a hands-on project to explore semantic search, RAG, LLM integration, and full-stack AI application development.

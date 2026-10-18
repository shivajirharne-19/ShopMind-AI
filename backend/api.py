
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.rag_pipeline import ask_shopmind


app = FastAPI(
    title="ShopMind AI API",
    description="Product search and RAG shopping assistant",
    version="1.0.0",
)


class ProductQuery(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_price: float | None = Field(default=None, gt=0)
    category: str | None = None
    min_ram_gb: float | None = Field(default=None, gt=0)
    min_storage_gb: float | None = Field(default=None, gt=0)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ShopMind AI",
    }


@app.post("/ask")
def ask_products(request: ProductQuery):
    try:
        result = ask_shopmind(
            query=request.query,
            top_k=request.top_k,
            max_price=request.max_price,
            category=request.category,
            min_ram_gb=request.min_ram_gb,
            min_storage_gb=request.min_storage_gb,
        )

        # Convert the Pandas DataFrame to JSON-safe records.
        products_json = json.loads(
            result["products"].to_json(orient="records")
        )

        return {
            "answer": result["answer"],
            "products": products_json,
            "count": len(products_json),
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="ShopMind AI failed to process the request.",
        )

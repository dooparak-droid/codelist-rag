"""api.py

FastAPI prediction service exposing REST endpoints for codelist generation,
health monitoring, and gold standard benchmark evaluation.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from codelist_rag.generate import CodelistPipeline
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.schemas import CodelistRequest, CodelistResponse, EvaluationRequest, EvaluationResponse
from codelist_rag.score import evaluate_codelist
from codelist_rag.terminology import TerminologyStore

# Configurable storage paths via environment variables
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
BM25_PATH = os.getenv("BM25_PATH", "./data/bm25_index.pkl")
TERMINOLOGY_JSON = os.getenv("TERMINOLOGY_JSON", "tests/fixtures/synthetic_terminology.json")

# Global pipeline instance initialized on startup
pipeline: CodelistPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager loading retrieval indexes on startup if present."""
    global pipeline
    retriever = None
    terminology = None

    if Path(CHROMA_PATH).exists() and Path(BM25_PATH).exists():
        try:
            retriever = HybridRetriever(chroma_path=CHROMA_PATH, bm25_path=BM25_PATH)
        except Exception as e:
            print(f"Warning: Could not load HybridRetriever on startup: {e}")

    if Path(TERMINOLOGY_JSON).exists():
        try:
            terminology = TerminologyStore.from_json(TERMINOLOGY_JSON)
        except Exception as e:
            print(f"Warning: Could not load TerminologyStore on startup: {e}")

    pipeline = CodelistPipeline(retriever=retriever, terminology=terminology)
    yield


app = FastAPI(
    title="SNOMED-CT Codelist RAG API",
    description="Retrieval-augmented LLM prediction service for clinical codelist generation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health and index status check."""
    has_retriever = pipeline is not None and pipeline.retriever is not None
    has_terminology = pipeline is not None and pipeline.terminology is not None
    return {
        "status": "healthy",
        "retriever_loaded": has_retriever,
        "terminology_loaded": has_terminology,
        "chroma_path": CHROMA_PATH,
        "bm25_path": BM25_PATH,
    }


@app.post("/codelist", response_model=CodelistResponse, status_code=status.HTTP_200_OK)
def generate_codelist(req: CodelistRequest):
    """
    Generate a SNOMED-CT codelist for a clinical condition.
    Retreats gracefully if index is unbuilt or model response fails to parse.
    """
    if req.use_rag and (pipeline is None or pipeline.retriever is None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index not loaded. Run 'codelist-rag build-index' before making RAG requests.",
        )

    assert pipeline is not None
    result = pipeline.generate(
        query=req.condition,
        strategy=req.strategy,
        use_rag=req.use_rag,
        n_results=req.n_results,
        provider=req.provider,
        model=req.model,
        base_url=req.base_url,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model failed to generate a valid codelist: {result['error']}",
        )

    return result


@app.post("/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
def evaluate_endpoint(req: EvaluationRequest):
    """Score a generated codelist against a gold standard reference set."""
    gold_set = set(req.gold_codes)
    retrieved_set = set(req.retrieved_codes)
    results = evaluate_codelist(
        generated_list=req.generated_codes,
        gold_codes=gold_set,
        retrieval_type=req.retrieval_type,
        retrieved_codes=retrieved_set,
    )
    return results

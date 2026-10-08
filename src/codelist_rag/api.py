"""api.py

FastAPI prediction service exposing REST endpoints for codelist generation,
health monitoring, and gold standard benchmark evaluation.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from codelist_rag import __version__
from codelist_rag.generate import DEFAULT_MODEL, DEFAULT_PROVIDER, CodelistPipeline
from codelist_rag.retrieve import HybridRetriever
from codelist_rag.schemas import CodelistRequest, CodelistResponse, EvaluationRequest, EvaluationResponse
from codelist_rag.score import evaluate_codelist
from codelist_rag.terminology import TerminologyStore

# Configurable storage paths via environment variables
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
BM25_PATH = os.getenv("BM25_PATH", "./data/bm25_index.pkl")
TERMINOLOGY_JSON = os.getenv("TERMINOLOGY_JSON")  # optional; used only when no index is loaded
TERMINOLOGY_RELEASE = os.getenv("TERMINOLOGY_RELEASE", "not recorded")

# Environment variable each hosted provider needs; Ollama needs none
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

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

    # With an index loaded, the fabrication check uses the index's own concept IDs.
    # A JSON file is used only when there is no index.
    if retriever is not None:
        try:
            terminology = TerminologyStore.from_chroma(retriever.collection)
        except Exception as e:
            print(f"Warning: Could not load TerminologyStore from index: {e}")
    elif TERMINOLOGY_JSON and Path(TERMINOLOGY_JSON).exists():
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
        "version": __version__,
        "default_model": f"{DEFAULT_PROVIDER}:{DEFAULT_MODEL}",
        "terminology_release": TERMINOLOGY_RELEASE,
        "terminology_concepts": len(pipeline.terminology) if has_terminology else 0,
        "retriever_loaded": has_retriever,
        "terminology_loaded": has_terminology,
        "chroma_path": CHROMA_PATH,
        "bm25_path": BM25_PATH,
    }


@app.post("/codelist", response_model=CodelistResponse, status_code=status.HTTP_200_OK)
def generate_codelist(req: CodelistRequest):
    """
    Generate a SNOMED-CT codelist for a clinical condition.
    Returns 503 if the index is unbuilt and 502 if the model response cannot be parsed.
    """
    if req.use_rag and (pipeline is None or pipeline.retriever is None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index not loaded. Run 'codelist-rag build-index' before making RAG requests.",
        )

    key_var = API_KEY_VARS.get(req.provider)
    if key_var and not os.getenv(key_var):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{key_var} is not set. Set it in the environment or a .env file to use provider '{req.provider}'.",
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

    return {**result, "version": __version__}


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

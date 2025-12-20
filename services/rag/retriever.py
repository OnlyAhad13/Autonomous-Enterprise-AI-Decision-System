"""
RAG Retriever Service.

Provides retrieval functionality and FastAPI endpoints for querying
the vector index.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query

# Local imports
from .ingest_vectors import (
    FAISSIndex,
    EmbeddingGenerator,
    Chunk,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
)

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Pydantic Models
# ============================================================================


class RetrieveRequest(BaseModel):
    """Request model for retrieval endpoint."""
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum similarity score")


class DocumentResult(BaseModel):
    """A single retrieved document."""
    chunk_id: str
    doc_id: str
    content: str
    source: str
    score: float
    timestamp: str
    metadata: Dict[str, Any] = {}


class RetrieveResponse(BaseModel):
    """Response model for retrieval endpoint."""
    query: str
    results: List[DocumentResult]
    num_results: int
    retrieved_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    index_loaded: bool
    num_documents: int


class IngestRequest(BaseModel):
    """Request for triggering ingestion."""
    docs_dir: Optional[str] = None
    force: bool = False


class IngestResponse(BaseModel):
    """Response after ingestion."""
    status: str
    num_chunks: int
    message: str


# ============================================================================
# Retriever Class
# ============================================================================


class Retriever:
    """Main retriever class for vector search."""
    
    _instance = None
    
    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL,
    ):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self._index: Optional[FAISSIndex] = None
        self._embedder: Optional[EmbeddingGenerator] = None
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> "Retriever":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load(self) -> bool:
        """Load the index and embedder."""
        if self._loaded:
            return True
        
        try:
            # Initialize embedder
            self._embedder = EmbeddingGenerator(self.model_name)
            
            # Load index
            self._index = FAISSIndex(dimension=self._embedder.dimension)
            self._index.load(self.index_dir)
            
            self._loaded = True
            logger.info(f"Retriever loaded with {len(self._index.chunks)} chunks")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Index not found at {self.index_dir}. Run ingestion first.")
            return False
        except Exception as e:
            logger.error(f"Error loading retriever: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[tuple]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            min_score: Minimum similarity score threshold.
            
        Returns:
            List of (Chunk, score) tuples.
        """
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Index not loaded. Run ingestion first.")
        
        # Generate query embedding
        query_embedding = self._embedder.embed_query(query)
        
        # Search
        results = self._index.search(query_embedding, top_k)
        
        # Filter by score if specified
        if min_score is not None:
            results = [(chunk, score) for chunk, score in results if score >= min_score]
        
        return results
    
    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        if self._index is None:
            return 0
        return len(self._index.chunks)
    
    @property
    def is_loaded(self) -> bool:
        """Whether the index is loaded."""
        return self._loaded


# ============================================================================
# Module-level retrieve function
# ============================================================================


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: Optional[float] = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Top-level retrieve function.
    
    Args:
        query: Search query text.
        top_k: Number of results to return.
        min_score: Minimum similarity score threshold.
        index_dir: Path to the index directory.
        model_name: Sentence transformer model name.
        
    Returns:
        List of dictionaries with document info and scores.
    """
    retriever = Retriever(index_dir=index_dir, model_name=model_name)
    retriever.load()
    
    results = retriever.retrieve(query, top_k, min_score)
    
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "source": chunk.source,
            "score": score,
            "timestamp": chunk.timestamp,
            "metadata": chunk.metadata,
        }
        for chunk, score in results
    ]


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="RAG Retriever API",
    description="Retrieval-Augmented Generation retriever service",
    version="1.0.0",
)


# Global retriever instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever.get_instance()
        _retriever.load()
    return _retriever


@app.on_event("startup")
async def startup_event():
    """Load index on startup."""
    logger.info("Starting RAG Retriever API...")
    retriever = get_retriever()
    if retriever.is_loaded:
        logger.info(f"Index loaded with {retriever.num_documents} chunks")
    else:
        logger.warning("Index not loaded. Ingest documents first.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    retriever = get_retriever()
    return HealthResponse(
        status="healthy",
        service="rag-retriever",
        index_loaded=retriever.is_loaded,
        num_documents=retriever.num_documents,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Retrieve relevant documents for a query.
    
    Returns top-k documents with similarity scores.
    """
    retriever = get_retriever()
    
    if not retriever.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first: python -m services.rag.ingest_vectors"
        )
    
    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        
        documents = [
            DocumentResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                source=chunk.source,
                score=score,
                timestamp=chunk.timestamp,
                metadata=chunk.metadata,
            )
            for chunk, score in results
        ]
        
        return RetrieveResponse(
            query=request.query,
            results=documents,
            num_results=len(documents),
            retrieved_at=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve", response_model=RetrieveResponse)
async def retrieve_get(
    query: str = Query(..., description="Query text"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results"),
):
    """GET endpoint for simple retrieval queries."""
    request = RetrieveRequest(query=query, top_k=top_k)
    return await retrieve_endpoint(request)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """Trigger document ingestion."""
    from .ingest_vectors import VectorIngester, DEFAULT_DOCS_DIR
    
    docs_dir = Path(request.docs_dir) if request.docs_dir else DEFAULT_DOCS_DIR
    
    try:
        ingester = VectorIngester(docs_dir=docs_dir)
        num_chunks = ingester.ingest()
        
        # Reload retriever
        global _retriever
        _retriever = None
        get_retriever()
        
        return IngestResponse(
            status="success",
            num_chunks=num_chunks,
            message=f"Ingested {num_chunks} chunks from {docs_dir}",
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLI
# ============================================================================


def main():
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "services.rag.retriever:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
    )


if __name__ == "__main__":
    main()

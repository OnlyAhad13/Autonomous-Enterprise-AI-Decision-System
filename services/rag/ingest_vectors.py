"""
RAG Vector Ingestion Pipeline.

Reads documents, generates embeddings, and indexes to vector databases.
Supports FAISS (local) and Milvus (production).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Iterator
from abc import ABC, abstractmethod

import numpy as np

# Lazy imports to handle optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "docs"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "vector_index"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Document:
    """Represents a document with content and metadata."""
    doc_id: str
    content: str
    source: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a document chunk with embedding."""
    chunk_id: str
    doc_id: str
    content: str
    source: str
    timestamp: str
    start_idx: int
    end_idx: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "metadata": self.metadata,
        }


# ============================================================================
# Document Loading
# ============================================================================


class DocumentLoader:
    """Load documents from various file formats."""
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}
    
    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
    
    def load_all(self) -> List[Document]:
        """Load all documents from the directory."""
        documents = []
        
        if not self.docs_dir.exists():
            logger.warning(f"Documents directory not found: {self.docs_dir}")
            return documents
        
        for filepath in self.docs_dir.rglob("*"):
            if filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self._load_file(filepath)
                if doc:
                    documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {self.docs_dir}")
        return documents
    
    def _load_file(self, filepath: Path) -> Optional[Document]:
        """Load a single file."""
        try:
            content = filepath.read_text(encoding="utf-8")
            
            # For JSON files, extract text content
            if filepath.suffix.lower() == ".json":
                data = json.loads(content)
                if isinstance(data, dict):
                    content = data.get("content", data.get("text", json.dumps(data)))
                elif isinstance(data, list):
                    content = "\n".join(str(item) for item in data)
            
            doc_id = hashlib.md5(str(filepath).encode()).hexdigest()[:12]
            
            return Document(
                doc_id=doc_id,
                content=content,
                source=str(filepath.relative_to(self.docs_dir)),
                timestamp=datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                metadata={"filename": filepath.name, "extension": filepath.suffix},
            )
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None


# ============================================================================
# Text Chunking
# ============================================================================


class TextChunker:
    """Chunk documents into smaller pieces with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document into chunks."""
        text = doc.content
        chunks = []
        
        # Split by separator first
        paragraphs = text.split(self.separator)
        
        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator
                current_chunk += para
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        doc, current_chunk, current_start, chunk_idx
                    ))
                    chunk_idx += 1
                    
                    # Apply overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + (self.separator if overlap_text else "") + para
                else:
                    current_chunk = para
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(self._create_chunk(doc, current_chunk, current_start, chunk_idx))
        
        return chunks
    
    def _create_chunk(
        self,
        doc: Document,
        content: str,
        start_idx: int,
        chunk_idx: int,
    ) -> Chunk:
        """Create a chunk object."""
        return Chunk(
            chunk_id=f"{doc.doc_id}_{chunk_idx}",
            doc_id=doc.doc_id,
            content=content.strip(),
            source=doc.source,
            timestamp=doc.timestamp,
            start_idx=start_idx,
            end_idx=start_idx + len(content),
            metadata={**doc.metadata, "chunk_index": chunk_idx},
        )


# ============================================================================
# Embedding Generation
# ============================================================================


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query])[0]


# ============================================================================
# Vector Index Base Class
# ============================================================================


class VectorIndex(ABC):
    """Abstract base class for vector indices."""
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings and chunks to index."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple]:
        """Search for similar vectors. Returns (chunk, score) tuples."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk."""
        pass


# ============================================================================
# FAISS Index
# ============================================================================


class FAISSIndex(VectorIndex):
    """FAISS-based vector index for local use."""
    
    def __init__(self, dimension: int = 384):
        if not HAS_FAISS:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized)
        self.chunks: List[Chunk] = []
        self._normalize = True
    
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings to FAISS index."""
        if len(embeddings) == 0:
            return
        
        # Normalize for cosine similarity
        if self._normalize:
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index (total: {len(self.chunks)})")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        """Search FAISS index."""
        if len(self.chunks) == 0:
            return []
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        if self._normalize:
            faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save FAISS index and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save chunks metadata
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save config
        config = {
            "dimension": self.dimension,
            "num_chunks": len(self.chunks),
            "created_at": datetime.now().isoformat(),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: Path) -> None:
        """Load FAISS index and metadata."""
        path = Path(path)
        
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Index not found at {path}")
        
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        with open(path / "chunks.json") as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                content=c["content"],
                source=c["source"],
                timestamp=c["timestamp"],
                start_idx=c["start_idx"],
                end_idx=c["end_idx"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]
        
        logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks from {path}")


# ============================================================================
# Milvus Connector (Example Implementation)
# ============================================================================


class MilvusConnector(VectorIndex):
    """
    Milvus connector for production vector storage.
    
    This is an example implementation. For production use, install pymilvus:
        pip install pymilvus
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "documents",
        dimension: int = 384,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self._connected = False
        self._client = None
        self._collection = None
        self.chunks: List[Chunk] = []
    
    def connect(self) -> bool:
        """Connect to Milvus server."""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            connections.connect("default", host=self.host, port=self.port)
            
            # Create collection if not exists
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(fields, description="Document chunks")
                self._collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                self._collection.create_index("embedding", index_params)
            else:
                self._collection = Collection(self.collection_name)
            
            self._collection.load()
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
            
        except ImportError:
            logger.warning("pymilvus not installed. Milvus connector disabled.")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add embeddings to Milvus."""
        if not self._connected:
            if not self.connect():
                logger.warning("Milvus not connected. Storing locally.")
                self.chunks.extend(chunks)
                return
        
        chunk_ids = [c.chunk_id for c in chunks]
        embeddings_list = embeddings.tolist()
        
        self._collection.insert([chunk_ids, embeddings_list])
        self._collection.flush()
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to Milvus")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        """Search Milvus collection."""
        if not self._connected:
            logger.warning("Milvus not connected")
            return []
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id"],
        )
        
        output = []
        for hits in results:
            for hit in hits:
                chunk_id = hit.entity.get("chunk_id")
                chunk = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
                if chunk:
                    output.append((chunk, hit.score))
        
        return output
    
    def save(self, path: Path) -> None:
        """Save metadata (Milvus handles vector storage)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(path / "chunks_metadata.json", "w") as f:
            json.dump(chunks_data, f, indent=2)
        
        config = {
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
        }
        with open(path / "milvus_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load metadata and connect to Milvus."""
        path = Path(path)
        
        with open(path / "chunks_metadata.json") as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                content=c["content"],
                source=c["source"],
                timestamp=c["timestamp"],
                start_idx=c["start_idx"],
                end_idx=c["end_idx"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]
        
        self.connect()


# ============================================================================
# Main Ingestion Class
# ============================================================================


class VectorIngester:
    """Main class for document ingestion and indexing."""
    
    def __init__(
        self,
        docs_dir: Path = DEFAULT_DOCS_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_milvus: bool = False,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
    ):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        
        self.loader = DocumentLoader(self.docs_dir)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator(model_name)
        
        # Initialize index
        if use_milvus:
            self.index = MilvusConnector(
                host=milvus_host,
                port=milvus_port,
                dimension=self.embedder.dimension,
            )
        else:
            self.index = FAISSIndex(dimension=self.embedder.dimension)
    
    def ingest(self) -> int:
        """Run the full ingestion pipeline."""
        logger.info("Starting document ingestion...")
        
        # Load documents
        documents = self.loader.load_all()
        if not documents:
            logger.warning("No documents found")
            return 0
        
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed(texts)
        
        # Index
        self.index.add(embeddings, all_chunks)
        
        # Save
        self.index.save(self.index_dir)
        
        logger.info(f"Ingestion complete. Index saved to {self.index_dir}")
        return len(all_chunks)
    
    def query(self, query: str, top_k: int = 5) -> List[tuple]:
        """Query the index."""
        query_embedding = self.embedder.embed_query(query)
        return self.index.search(query_embedding, top_k)


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into vector index")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Directory containing documents",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory to save index",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--use-milvus",
        action="store_true",
        help="Use Milvus instead of FAISS",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Test query after ingestion",
    )
    
    args = parser.parse_args()
    
    ingester = VectorIngester(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_milvus=args.use_milvus,
    )
    
    num_chunks = ingester.ingest()
    print(f"\n✓ Ingested {num_chunks} chunks")
    
    if args.query:
        print(f"\nQuery: {args.query}")
        results = ingester.query(args.query, top_k=3)
        for i, (chunk, score) in enumerate(results):
            print(f"\n[{i+1}] Score: {score:.4f}")
            print(f"    Source: {chunk.source}")
            print(f"    Content: {chunk.content[:200]}...")


if __name__ == "__main__":
    main()

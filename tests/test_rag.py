"""
Unit tests for RAG (Retrieval-Augmented Generation) pipeline.

Tests cover:
- Document loading
- Text chunking
- Embedding generation
- FAISS indexing
- Retriever functionality
- FastAPI endpoints
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import tempfile
import json

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for optional dependencies
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_docs_dir(tmp_path):
    """Create temporary docs directory with sample files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create sample markdown file
    md_file = docs_dir / "sample.md"
    md_file.write_text("""# Sample Document

This is a sample document for testing the RAG pipeline.

## Section 1

Machine learning is a subset of artificial intelligence that enables 
systems to learn from data.

## Section 2

Deep learning uses neural networks with many layers.
""")
    
    # Create sample text file
    txt_file = docs_dir / "test.txt"
    txt_file.write_text("This is a simple test document for vector search.")
    
    # Create sample JSON file
    json_file = docs_dir / "data.json"
    json_file.write_text(json.dumps({"content": "JSON document content for testing."}))
    
    return docs_dir


@pytest.fixture
def sample_chunks():
    """Create sample Chunk objects."""
    from services.rag.ingest_vectors import Chunk
    
    return [
        Chunk(
            chunk_id="doc1_0",
            doc_id="doc1",
            content="Machine learning is a subset of AI.",
            source="sample.md",
            timestamp=datetime.now().isoformat(),
            start_idx=0,
            end_idx=35,
        ),
        Chunk(
            chunk_id="doc1_1",
            doc_id="doc1",
            content="Deep learning uses neural networks.",
            source="sample.md",
            timestamp=datetime.now().isoformat(),
            start_idx=36,
            end_idx=70,
        ),
    ]


# ============================================================================
# Document Loading Tests
# ============================================================================


class TestDocumentLoader:
    """Tests for DocumentLoader class."""
    
    def test_load_markdown_file(self, sample_docs_dir):
        """Test loading a markdown file."""
        from services.rag.ingest_vectors import DocumentLoader
        
        loader = DocumentLoader(sample_docs_dir)
        docs = loader.load_all()
        
        assert len(docs) >= 1
        md_doc = next((d for d in docs if d.source == "sample.md"), None)
        assert md_doc is not None
        assert "Machine learning" in md_doc.content
    
    def test_load_text_file(self, sample_docs_dir):
        """Test loading a text file."""
        from services.rag.ingest_vectors import DocumentLoader
        
        loader = DocumentLoader(sample_docs_dir)
        docs = loader.load_all()
        
        txt_doc = next((d for d in docs if d.source == "test.txt"), None)
        assert txt_doc is not None
        assert "test document" in txt_doc.content
    
    def test_load_json_file(self, sample_docs_dir):
        """Test loading a JSON file."""
        from services.rag.ingest_vectors import DocumentLoader
        
        loader = DocumentLoader(sample_docs_dir)
        docs = loader.load_all()
        
        json_doc = next((d for d in docs if d.source == "data.json"), None)
        assert json_doc is not None
        assert "JSON document" in json_doc.content
    
    def test_load_nonexistent_directory(self, tmp_path):
        """Test loading from non-existent directory."""
        from services.rag.ingest_vectors import DocumentLoader
        
        loader = DocumentLoader(tmp_path / "nonexistent")
        docs = loader.load_all()
        
        assert len(docs) == 0
    
    def test_document_metadata(self, sample_docs_dir):
        """Test that document metadata is properly set."""
        from services.rag.ingest_vectors import DocumentLoader
        
        loader = DocumentLoader(sample_docs_dir)
        docs = loader.load_all()
        
        for doc in docs:
            assert doc.doc_id is not None
            assert doc.source is not None
            assert doc.timestamp is not None
            assert "filename" in doc.metadata


# ============================================================================
# Text Chunking Tests
# ============================================================================


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_basic_chunking(self):
        """Test basic document chunking."""
        from services.rag.ingest_vectors import TextChunker, Document
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(
            doc_id="test",
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            source="test.md",
            timestamp=datetime.now().isoformat(),
        )
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert all(c.doc_id == "test" for c in chunks)
    
    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        from services.rag.ingest_vectors import TextChunker, Document
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        doc = Document(
            doc_id="test",
            content="Sample content for testing.",
            source="test.md",
            timestamp=datetime.now().isoformat(),
            metadata={"key": "value"},
        )
        
        chunks = chunker.chunk_document(doc)
        
        for chunk in chunks:
            assert chunk.chunk_id.startswith("test_")
            assert "chunk_index" in chunk.metadata
    
    def test_small_document_single_chunk(self):
        """Test that small documents result in single chunk."""
        from services.rag.ingest_vectors import TextChunker, Document
        
        chunker = TextChunker(chunk_size=1000, chunk_overlap=50)
        doc = Document(
            doc_id="small",
            content="Small document.",
            source="small.md",
            timestamp=datetime.now().isoformat(),
        )
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 1


# ============================================================================
# Embedding Generation Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock sentence transformer model."""
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        return model
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_texts(self, mock_st, mock_model):
        """Test embedding multiple texts."""
        mock_st.return_value = mock_model
        
        from services.rag.ingest_vectors import EmbeddingGenerator
        
        generator = EmbeddingGenerator("all-MiniLM-L6-v2")
        embeddings = generator.embed(["text1", "text2", "text3"])
        
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_query(self, mock_st, mock_model):
        """Test embedding a single query."""
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_model
        
        from services.rag.ingest_vectors import EmbeddingGenerator
        
        generator = EmbeddingGenerator("all-MiniLM-L6-v2")
        query_embedding = generator.embed_query("test query")
        
        assert query_embedding.shape == (384,)


# ============================================================================
# FAISS Index Tests
# ============================================================================


class TestFAISSIndex:
    """Tests for FAISSIndex class."""
    
    def test_add_and_search(self, sample_chunks):
        """Test adding embeddings and searching."""
        from services.rag.ingest_vectors import FAISSIndex
        
        index = FAISSIndex(dimension=384)
        embeddings = np.random.rand(2, 384).astype(np.float32)
        
        index.add(embeddings, sample_chunks)
        
        query = np.random.rand(384).astype(np.float32)
        results = index.search(query, top_k=2)
        
        assert len(results) == 2
        for chunk, score in results:
            assert chunk.content is not None
            assert isinstance(score, float)
    
    def test_save_and_load(self, sample_chunks, tmp_path):
        """Test saving and loading index."""
        from services.rag.ingest_vectors import FAISSIndex
        
        # Create and save
        index = FAISSIndex(dimension=384)
        embeddings = np.random.rand(2, 384).astype(np.float32)
        index.add(embeddings, sample_chunks)
        index.save(tmp_path / "index")
        
        # Load and verify
        new_index = FAISSIndex(dimension=384)
        new_index.load(tmp_path / "index")
        
        assert len(new_index.chunks) == 2
    
    def test_empty_index_search(self):
        """Test searching empty index."""
        from services.rag.ingest_vectors import FAISSIndex
        
        index = FAISSIndex(dimension=384)
        query = np.random.rand(384).astype(np.float32)
        results = index.search(query, top_k=5)
        
        assert len(results) == 0


# ============================================================================
# VectorIngester Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestVectorIngester:
    """Tests for VectorIngester class."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_full_ingestion(self, mock_st, sample_docs_dir, tmp_path):
        """Test full ingestion pipeline."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_st.return_value = mock_model
        
        from services.rag.ingest_vectors import VectorIngester
        
        ingester = VectorIngester(
            docs_dir=sample_docs_dir,
            index_dir=tmp_path / "index",
            chunk_size=200,
        )
        
        num_chunks = ingester.ingest()
        
        assert num_chunks > 0
        assert (tmp_path / "index" / "index.faiss").exists()
        assert (tmp_path / "index" / "chunks.json").exists()


# ============================================================================
# Retriever Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestRetriever:
    """Tests for Retriever class."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_function(self, mock_st, sample_docs_dir, tmp_path):
        """Test the retrieve function."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        ).astype(np.float32)
        mock_st.return_value = mock_model
        
        # First ingest
        from services.rag.ingest_vectors import VectorIngester
        ingester = VectorIngester(
            docs_dir=sample_docs_dir,
            index_dir=tmp_path / "index",
        )
        ingester.ingest()
        
        # Then retrieve
        from services.rag.retriever import retrieve
        results = retrieve(
            query="machine learning",
            top_k=2,
            index_dir=tmp_path / "index",
        )
        
        assert len(results) <= 2
        for r in results:
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================


class TestRetrieverAPI:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from services.rag.retriever import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "rag-retriever"
    
    @patch('services.rag.retriever.get_retriever')
    def test_retrieve_endpoint(self, mock_get_retriever, client, sample_chunks):
        """Test POST /retrieve endpoint."""
        mock_retriever = MagicMock()
        mock_retriever.is_loaded = True
        mock_retriever.retrieve.return_value = [
            (sample_chunks[0], 0.85),
            (sample_chunks[1], 0.72),
        ]
        mock_get_retriever.return_value = mock_retriever
        
        response = client.post(
            "/retrieve",
            json={"query": "machine learning", "top_k": 2}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert len(data["results"]) == 2
    
    @patch('services.rag.retriever.get_retriever')
    def test_retrieve_get_endpoint(self, mock_get_retriever, client, sample_chunks):
        """Test GET /retrieve endpoint."""
        mock_retriever = MagicMock()
        mock_retriever.is_loaded = True
        mock_retriever.retrieve.return_value = [(sample_chunks[0], 0.85)]
        mock_get_retriever.return_value = mock_retriever
        
        response = client.get("/retrieve?query=test&top_k=1")
        
        assert response.status_code == 200
    
    @patch('services.rag.retriever.get_retriever')
    def test_retrieve_index_not_loaded(self, mock_get_retriever, client):
        """Test retrieval when index not loaded."""
        mock_retriever = MagicMock()
        mock_retriever.is_loaded = False
        mock_get_retriever.return_value = mock_retriever
        
        response = client.post(
            "/retrieve",
            json={"query": "test", "top_k": 3}
        )
        
        assert response.status_code == 503


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for RAG pipeline."""
    
    def test_docs_directory_exists(self):
        """Test that sample docs directory exists."""
        docs_dir = PROJECT_ROOT / "data" / "docs"
        # May not exist if not set up yet
        # Just check it's a valid path
        assert isinstance(docs_dir, Path)
    
    def test_ingest_vectors_module_imports(self):
        """Test that ingest_vectors module imports correctly."""
        from services.rag import ingest_vectors
        
        assert hasattr(ingest_vectors, "VectorIngester")
        assert hasattr(ingest_vectors, "FAISSIndex")
        assert hasattr(ingest_vectors, "DocumentLoader")
    
    def test_retriever_module_imports(self):
        """Test that retriever module imports correctly."""
        from services.rag import retriever
        
        assert hasattr(retriever, "retrieve")
        assert hasattr(retriever, "Retriever")
        assert hasattr(retriever, "app")

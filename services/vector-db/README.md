# Vector Database Service

High-dimensional vector storage and similarity search.

## Overview

This service manages:
- Vector embedding storage
- Similarity search at scale
- Hybrid search (vector + metadata)
- Index management and optimization

## Structure

```
vector-db/
├── embeddings/          # Embedding generation
│   ├── __init__.py
│   └── generator.py
├── indexes/             # Index configurations
│   ├── __init__.py
│   └── index_config.py
├── queries/             # Query templates
│   ├── __init__.py
│   └── search.py
├── sync/                # Data synchronization
│   └── sync_pipeline.py
├── config.py
└── README.md
```

## Indexes

| Index | Dimensions | Metric | Use Case |
|-------|------------|--------|----------|
| `documents` | 1536 | Cosine | Document retrieval |
| `entities` | 768 | Euclidean | Entity matching |
| `queries` | 1536 | Dot Product | Query understanding |

## Usage

```python
from vector_db import VectorStore

store = VectorStore()

# Index embeddings
store.upsert(
    index="documents",
    vectors=[
        {"id": "doc_1", "values": [...], "metadata": {"source": "kb"}}
    ]
)

# Similarity search
results = store.query(
    index="documents",
    vector=query_embedding,
    top_k=10,
    filter={"source": "kb"}
)
```

## Supported Backends

| Backend | Status | Notes |
|---------|--------|-------|
| Pinecone | ✅ Supported | Managed, serverless |
| Milvus | ✅ Supported | Self-hosted, scalable |
| ChromaDB | ✅ Supported | Local development |
| Qdrant | 🔄 Planned | Self-hosted alternative |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_DB_BACKEND` | Backend to use | `chromadb` |
| `PINECONE_API_KEY` | Pinecone API key | - |
| `MILVUS_HOST` | Milvus server host | `localhost` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |

## Development

```bash
# Start ChromaDB locally
docker run -p 8000:8000 chromadb/chroma

# Sync embeddings
poetry run python sync/sync_pipeline.py

# Run search tests
poetry run pytest tests/test_search.py
```

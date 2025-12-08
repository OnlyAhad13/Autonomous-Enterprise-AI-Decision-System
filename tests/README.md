# Tests

Test suite for the Autonomous Enterprise AI Decision System.

## Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_data_ingestion/
│   ├── test_data_processing/
│   ├── test_feature_store/
│   ├── test_ml_platform/
│   ├── test_serving/
│   ├── test_vector_db/
│   └── test_agent/
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   └── test_api.py
├── e2e/                     # End-to-end tests
│   └── test_decision_flow.py
├── fixtures/                # Test fixtures
│   └── sample_data.json
├── conftest.py              # Pytest configuration
└── README.md
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=services --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_serving/test_api.py

# Run marked tests
poetry run pytest -m "not slow"
```

## Test Categories

| Marker | Description |
|--------|-------------|
| `unit` | Fast unit tests |
| `integration` | Integration tests (require services) |
| `e2e` | End-to-end tests |
| `slow` | Long-running tests |

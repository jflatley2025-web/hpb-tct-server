# HPB-TCT Test Suite

This directory contains the test suite for the HPB-TCT trading system.

## Structure

```
tests/
├── unit/                   # Unit tests for individual modules
│   ├── test_range_scanner.py
│   └── test_hpb_rig_validator.py
├── integration/           # Integration tests for multi-component workflows
├── fixtures/              # Test data and mock responses
│   └── mock_candles.json
├── conftest.py           # Shared pytest fixtures
└── README.md             # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

### Run specific test file
```bash
pytest tests/unit/test_range_scanner.py -v
```

### Run tests by marker
```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m "not slow"    # Skip slow tests
```

### Run tests in parallel (faster)
```bash
pip install pytest-xdist
pytest -n auto
```

## Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests involving multiple components
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.requires_api` - Tests requiring external API access

## Writing Tests

### Unit Test Example
```python
@pytest.mark.unit
def test_my_function():
    result = my_function(input_data)
    assert result == expected_value
```

### Async Test Example
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Using Fixtures
```python
def test_with_fixture(mock_context_basic):
    # mock_context_basic comes from conftest.py
    result = validator(mock_context_basic)
    assert result["status"] == "VALID"
```

## Coverage Goals

| Module | Target Coverage | Current Coverage |
|--------|----------------|------------------|
| range_scanner.py | 80%+ | ✅ Comprehensive |
| hpb_rig_validator.py | 80%+ | ✅ Comprehensive |
| tct_model_detector.py | 80%+ | ⏳ Pending |
| server_mexc.py | 75%+ | ⏳ Pending |
| risk_model.py | 70%+ | ⏳ Pending |

## CI/CD

Tests run automatically on:
- Every push to `main`, `master`, `develop`, or `claude/**` branches
- Every pull request
- Python versions: 3.10, 3.11

View test results in GitHub Actions tab.

## Adding New Tests

1. Create test file: `tests/unit/test_<module_name>.py`
2. Import module and required fixtures
3. Write test classes/functions with descriptive names
4. Add appropriate markers (@pytest.mark.unit, etc.)
5. Run tests locally before committing
6. Ensure coverage doesn't decrease

## Mock Data

Mock API responses and test data are stored in `tests/fixtures/`. Add new fixtures when testing external API integrations.

## Troubleshooting

### Tests failing locally?
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv
```

### Need to debug a test?
```bash
# Run with print statements visible
pytest -s

# Drop into debugger on failure
pytest --pdb
```

## Next Steps

Priority modules to add tests for:
1. `tct_model_detector.py` - Critical trading logic
2. `server_mexc.py` - API endpoints and gate validation
3. `risk_model.py` - Risk calculation algorithms
4. `telegram_bot.py` - Bot command handlers
5. `hpb_nvs_module.py` - Sentiment analysis

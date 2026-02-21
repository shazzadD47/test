## delinateai API

## Run using Docker

first clone the project from github

```
git clone https://github.com/Liberate-Labs/delineate-ai.git
```

go inside the folder and install the requirments packages.

```
cd delineate-ai
```

then run

```
docker compose -f docker-compose-dev.yml up --build
```

### install

first clone the project from github

```
git clone https://github.com/Liberate-Labs/delineate-ai.git
```

go inside the folder and install the requirments packages.

```
cd delineate-ai
```

Make a virtual environment and activate it

```
python -m venv venv

venv\Scripts\activate.bat
```

```
pip install -r requirements.txt
```

start the uvicorn server

```
uvicorn app.main:app
```

## Development

### Install the required packages

```bash
pip install -r requirements.dev
```

### Install pre-commit hooks

```bash
pre-commit install

# to manually run the hooks
make pre-commit
```

### Checking logs for individual services:

```
docker compose logs -f io-celery --tail 100
```

## Testing

### Running Tests

To run the test suite, use pytest:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=app
```

To run tests in a specific directory:

```bash
pytest tests/core/
```

To run a specific test file:

```bash
pytest tests/core/test_crud.py
```

To run tests with verbose output:

```bash
pytest -v
```

### Test Structure

Tests are organized in the `tests/` directory with the following structure:
- `tests/core/` - Tests for core functionality
- `tests/endpoints/` - Tests for API endpoints
- `conftest.py` - Shared test fixtures and configuration

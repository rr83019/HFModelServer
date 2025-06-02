## Overview

This document provides detailed documentation for the Sentence Embedding API application. It explains the architecture, components, installation steps, configuration options, usage examples, and deployment instructions.

---

## Table of Contents

1. [Application Overview](#application-overview)
2. [Architecture and Components](#architecture-and-components)

   * [Directory Structure](#directory-structure)
   * [Key Components](#key-components)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)

   * [Python Environment](#python-environment)
   * [Docker Setup](#docker-setup)
5. [Configuration](#configuration)
6. [Running the Application Locally](#running-the-application-locally)
7. [API Endpoints](#api-endpoints)

   * [Health Check](#health-check)
   * [Train](#train)
   * [Embed](#embed)
   * [Search](#search)
8. [Request and Response Examples](#request-and-response-examples)
9. [Logging](#logging)
10. [Docker Deployment](#docker-deployment)
11. [Troubleshooting](#troubleshooting)
12. [Extending the Application](#extending-the-application)
13. [Appendix](#appendix)

---

## Application Overview

The Sentence Embedding API is a FastAPI-based microservice that provides embeddings for input sentences using a pre-trained transformer model (`sentence-transformers/all-MiniLM-L6-v2`). It leverages FAISS (Facebook AI Similarity Search) for indexing and approximate nearest neighbor (ANN) search. Key functionalities include:

* **Encode**: Convert sentences into fixed-size embedding vectors.
* **Train**: Build or update a FAISS index from encoded embeddings.
* **Embed**: Generate embeddings for sentences and add them to the FAISS index.
* **Search**: Query the FAISS index to retrieve top-k nearest neighbors for a given query sentence.
* **Health Check**: Verify that the service is up and running.

This documentation covers all aspects required to understand, install, configure, and use the service.

---

## Architecture and Components

### Directory Structure

```plaintext
.
├── src/
│   ├── main.py
│   ├── model.py
│   ├── storage.py
│   └── schema.py
├── requirements.txt
├── Dockerfile
└── README.md  (this document)
```

### Key Components

1. **src/main.py**

   * Entry point for the FastAPI application.
   * Defines API endpoints (`/health`, `/train`, `/embed`, `/search`).
   * Sets up model and FAISS index during application lifespan.
   * Handles request validation and error responses.

2. **src/model.py (SentenceEmbeddingModel)**

   * Wraps the HuggingFace `SentenceTransformer` model.
   * Provides an asynchronous method `encode()` to convert sentences into embeddings.
   * Implements `close()` to free resources (e.g., GPU memory).

3. **src/storage.py (FaissIndex)**

   * Thread-safe wrapper around a FAISS `IndexIVFFlat`.
   * Methods: `train()`, `add()`, `search()`, `save()`, `load()`.
   * Ensures only one thread at a time can modify or persist the index.

4. **src/schema.py**

   * Pydantic models for request and response schemas.
   * `EmbeddingRequest`, `EmbeddingResponse`, `SearchRequest`, `SearchResponse`.

5. **requirements.txt**

   * Lists Python package dependencies, e.g., FastAPI, Uvicorn, sentence-transformers, faiss-cpu, numpy, torch.

6. **Dockerfile**

   * Defines a containerized environment for the application.
   * Installs system and Python dependencies.
   * Copies application code and sets the startup command.

---

## Prerequisites

Before installing and running the application, ensure you have the following:

* **Python 3.9+**
  (Install from [https://www.python.org/downloads/](https://www.python.org/downloads/))
* **pip** (Python package installer)
* **Git** (optional, for cloning the repository)
* **Docker & Docker Compose** (for containerized deployment)
* Adequate disk and memory resources (FAISS index and transformer model can consume several GBs).

---

## Installation

### Python Environment

1. **Clone the repository** (if applicable):

   ```bash
   git clone https://github.com/your-organization/sentence-embedding-api.git
   cd sentence-embedding-api
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:

   ```bash
   python -c "import fastapi; import faiss; import torch; print('Dependencies installed successfully!')"
   ```

### Docker Setup

1. **Install Docker**:

   * [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
2. **Build the Docker image**:

   ```bash
   docker build -t sentence-embedding-api .
   ```
3. **Run the container**:

   ```bash
   docker run -d --name embedding_api -p 8000:8000 sentence-embedding-api
   ```
4. **Stop and remove container**:

   ```bash
   docker stop embedding_api
   docker rm embedding_api
   ```

---

## Configuration

No environment variables are strictly required for basic usage. However, you may customize:

* **FAISS index directory**
  By default, on shutdown, the index will be saved to `./faiss_index/`.
  You can change this path by modifying `app/main.py` (in `lifespan` function).

* **Model name**
  To use a different HuggingFace model, update the default `model_name` in `SentenceEmbeddingModel.__init__()`.

* **FAISS parameters**
  Adjust `dim`, `nlist`, and `metric` when instantiating `FaissIndex` in `main.py`.

* **Logging level**
  Currently set to `INFO`. Modify `logging.basicConfig(level=logging.INFO)` to `DEBUG` or `WARNING` as needed.

---

## Running the Application Locally

1. **Activate your Python environment** (if not already active):

   ```bash
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

2. **Start the FastAPI server**:

   ```bash
   python src/main.py
   ```

   This uses the embedded Uvicorn call in `main.py`. The server will listen on `http://0.0.0.0:8000`.

3. **Verify**:

   * Open a browser and go to `http://localhost:8000/health`. You should see:

     ```json
     {"status": "healthy"}
     ```
   * Visit the Swagger UI at `http://localhost:8000/docs` to explore interactive documentation.

---

## API Endpoints

### Health Check

**GET** `/health`

* **Description**: Verifies that the service is up and running.
* **Response**: `{ "status": "healthy" }`
* **HTTP Status Codes**:

  * `200 OK` — Service is healthy.

---

### Train

**POST** `/train`

* **Description**: Trains the FAISS index on provided sentences.
* **Request Body** (`EmbeddingRequest`):

  ```json
  {
    "sentences": ["Sentence one.", "Another sentence to encode."]
  }
  ```
* **Response**:

  ```json
  { "message": "Training complete" }
  ```
* **HTTP Status Codes**:

  * `200 OK` — Training initiated/completed successfully.
  * `400 Bad Request` — No sentences provided.
  * `500 Internal Server Error` — Training failure (e.g., index I/O error).

**Behavior**:

1. Encodes sentences to embeddings via `SentenceEmbeddingModel.encode()`.
2. Calls `FaissIndex.train()` under a thread lock.
3. If the index was previously untrained, this builds the IVF clusters.
4. Returns immediately after training (no batching).

---

### Embed

**POST** `/embed`

* **Description**: Generates embeddings for input sentences and adds them to the FAISS index.
* **Request Body** (`EmbeddingRequest`):

  ```json
  {
    "sentences": ["Sample sentence.", "FastAPI is awesome!"]
  }
  ```
* **Response** (`EmbeddingResponse`):

  ```json
  {
    "embeddings": [
      [0.123, 0.456, ..., 0.789],
      [0.234, 0.567, ..., 0.890]
    ]
  }
  ```
* **HTTP Status Codes**:

  * `200 OK` — Embeddings generated and added to index.
  * `400 Bad Request` — No sentences provided.
  * `500 Internal Server Error` — Embedding or index addition failure.

**Behavior**:

1. Encodes sentences to embeddings via `SentenceEmbeddingModel.encode()`.
2. Calls `FaissIndex.add()` under a thread lock to store new vectors.
3. Returns the newly computed embeddings to the client.

---

### Search

**POST** `/search`

* **Description**: Searches the FAISS index and returns top-k nearest neighbors for a query sentence.
* **Request Body** (`SearchRequest`):

  ```json
  {
    "query": "How do I use FAISS?",
    "k": 5
  }
  ```
* **Response** (`SearchResponse`):

  ```json
  {
    "distances": [[0.123, 0.234, 0.345, 0.456, 0.567]],
    "indices": [[12, 45, 78, 3, 19]]
  }
  ```
* **HTTP Status Codes**:

  * `200 OK` — Search executed successfully.
  * `400 Bad Request` — No query provided.
  * `500 Internal Server Error` — Search failure (e.g., index not trained).

**Behavior**:

1. Encodes single query into embedding via `SentenceEmbeddingModel.encode()`.
2. Calls `FaissIndex.search()` (thread-safe for reads) to retrieve distances and indices.
3. Returns arrays of distances and indices to the client.

---

## Request and Response Examples

### 1. Health Check

**Request**:

```bash
curl -X GET http://localhost:8000/health
```

**Response**:

```json
{ "status": "healthy" }
```

### 2. Train Endpoint

**Request**:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["Hello world", "FastAPI demo"]}'
```

**Response**:

```json
{ "message": "Training complete" }
```

### 3. Embed Endpoint

**Request**:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["OpenAI is cool", "FAISS integration"]}'
```

**Response**:

```json
{
  "embeddings": [
    [0.12345, 0.67890, ..., 0.54321],
    [0.23456, 0.78901, ..., 0.65432]
  ]
}
```

### 4. Search Endpoint

**Request**:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "k": 3}'
```

**Response**:

```json
{
  "distances": [[0.123, 0.234, 0.345]],
  "indices": [[5, 23, 42]]
}
```

---

## Logging

* Logging is configured at the application and module level using Python's `logging` module.
* The root logger prints timestamps, log levels (INFO, DEBUG, WARNING, ERROR), the logger name, and the log message.
* **Levels used**:

  * `INFO` for startup/shutdown and major lifecycle events.
  * `DEBUG` for detailed step-by-step operations (e.g., number of sentences encoded).
  * `WARNING` and `ERROR` for invalid inputs and exceptions.

Log messages can be viewed in the console or redirected to a file by configuring `logging.basicConfig(...)` accordingly.

---

## Docker Deployment

1. **Build Docker Image**:

   ```bash
   docker build -t sentence-embedding-api .
   ```
2. **Run the Container**:

   ```bash
   docker run -d \
     --name embedding_api \
     -p 8000:8000 \
     sentence-embedding-api
   ```
3. **Verify**:

   ```bash
   curl http://localhost:8000/health
   ```
4. **Logs**:

   ```bash
   docker logs embedding_api
   ```
5. **Stop & Remove**:

   ```bash
   docker stop embedding_api
   docker rm embedding_api
   ```

**Dockerfile Highlights**:

* Based on `python:3.10-slim` image.
* Installs system dependencies (e.g., `libsm6`, `libxrender-dev`) required by `sentence-transformers`.
* Installs Python dependencies from `requirements.txt`.
* Copies application code and exposes port `8000`.
* Default command: `python main.py`, which internally starts Uvicorn.


---

## Troubleshooting

1. **Index not found or untrained**:

   * If `/search` returns an error stating "Index must be trained before searching", ensure you have called `/train` at least once with a non-empty list of sentences.

2. **CUDA out-of-memory**:

   * If you see CUDA memory errors, it may be due to large batches. Reduce batch sizes or switch to CPU by setting `device='cpu'` when loading the `SentenceTransformer` model.

3. **Missing libraries in Docker**:

   * If Docker logs show missing `libsm6` or `libxrender`, confirm that the `apt-get install` step in the Dockerfile included all required dependencies.

4. **Slow model loading**:

   * Transformer models can take several seconds to load. This happens at startup. Ensure your health check accounts for this delay.

5. **High memory usage**:

   * The FAISS index and transformer model can consume multiple gigabytes. Monitor memory usage (`htop`, `docker stats`) and consider using GPU or a larger instance if needed.

---

## Extending the Application

1. **Batching and Asynchronous I/O**:

   * Currently, each request encodes and processes embeddings immediately. For high throughput, implement batching logic or queue-based processing (e.g., using Redis or RabbitMQ).

2. **Persistent Storage and Index Versions**:

   * Instead of saving to a single directory on shutdown, version your index files (e.g., `faiss_index_v1.ivf`) and provide endpoints to switch between versions.

3. **Authentication and Authorization**:

   * Add OAuth2 or API key-based security to protect endpoints. Use FastAPI's `Security` utilities.

4. **Metrics and Monitoring**:

   * Integrate Prometheus, Grafana, or a logging aggregator (ELK/EFK) to monitor request rates, latency, and errors.

5. **Model Variants**:

   * Allow dynamic switching of transformer models via environment variables or API parameters.

6. **GPU Support**:

   * Modify `SentenceEmbeddingModel` to load on GPU if available. In `model.py`: `SentenceTransformer(model_name, device='cuda')`.

7. **Batch Search and Parallelism**:

   * Expose an endpoint to accept multiple queries at once and return results in a single response.
   * Use FAISS’s GPU index classes (`faiss.IndexIVFFlatGPU`) when running on GPU for faster searches.

---

## Appendix

### A. Environment Variables (Optional)

* `MODEL_NAME`

  * Description: Override default transformer model.
  * Example: `export MODEL_NAME=stsb-roberta-base`

* `FAISS_INDEX_PATH`

  * Description: Directory path to save/load FAISS index.
  * Example: `export FAISS_INDEX_PATH=/data/faiss_index`

* `LOG_LEVEL`

  * Description: Set application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
  * Example: `export LOG_LEVEL=DEBUG`

### B. Useful Commands

* **Rebuild Docker image (force fresh)**:

  ```bash
  docker build --no-cache -t sentence-embedding-api .
  ```

* **View container logs**:

  ```bash
  docker logs -f embedding_api
  ```

* **Enter running container (for troubleshooting)**:

  ```bash
  docker exec -it embedding_api /bin/bash
  ```

* **Check Python versions**:

  ```bash
  python --version
  pip show sentence-transformers
  pip show faiss-cpu
  ```

* **Run tests** (if tests are added in the future):

  ```bash
  pytest tests/
  ```

---


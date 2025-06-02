from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from model import SentenceEmbeddingModel
from routes import router
from storage import FaissIndex

import os

# Set environment variables to control threading behavior for arm (mac) architecture
os.environ["OMP_NUM_THREADS"] = "1"

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentence_embedding_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan function to handle startup and shutdown events.
    Initializes the SentenceEmbeddingModel and FaissIndex on startup,
    and ensures they are properly closed/saved on shutdown.
    """
    try:
        logger.info("Lifespan startup initiated")

        # Load and initialize the sentence embedding model
        model = SentenceEmbeddingModel()
        app.state.model = model
        logger.info("SentenceEmbeddingModel initialized successfully")

        # Load and initialize the Faiss index with specified dimension, nlist, and metric
        faiss_index = FaissIndex(dim=384, nlist=10, metric='L2')
        app.state.index = faiss_index
        logger.info("FaissIndex initialized successfully")

        # Yield control back to FastAPI
        yield 
    except Exception as e:
        # Log any exceptions that occur during startup
        logger.exception(f"Error during startup initialization: {e}")
        raise
    finally:
        # On shutdown, close the model and save the index to disk
        app.state.model.close()
        logger.info("SentenceEmbeddingModel shutdown complete")

        app.state.index.save("faiss_index/")
        logger.info("FaissIndex saved successfully to 'faiss_index/'")
        logger.info("Lifespan shutdown complete")


# Create the FastAPI application instance and pass in the custom lifespan handler
app = FastAPI(title="Sentence Embedding API", version="1.0", lifespan=lifespan)

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the application is running.
    Returns a simple JSON payload indicating health status.
    """
    logger.debug("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    """
    Entry point for running the application with Uvicorn.
    """
    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
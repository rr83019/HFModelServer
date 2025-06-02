from fastapi import HTTPException, APIRouter, Request, Depends
import logging
from model import SentenceEmbeddingModel
import faiss

from schema import EmbeddingRequest, EmbeddingResponse, SearchRequest, SearchResponse

router = APIRouter()
logger = logging.getLogger(__name__)

def get_model(request: Request) -> SentenceEmbeddingModel:
    return request.app.state.model

def get_faiss_index(request: Request) -> faiss.Index:
    return request.app.state.index

@router.post("/train")
async def train(
    request: EmbeddingRequest, 
    model: SentenceEmbeddingModel = Depends(get_model),
    index: faiss.Index = Depends(get_faiss_index)
):
    """
    Endpoint to train the Faiss index on a batch of sentences.
    - Encodes input sentences using the SentenceEmbeddingModel.
    - Trains the Faiss index on the resulting embeddings.
    """
    
    # Validate that the request contains sentences
    if not request.sentences:
        logger.warning("No sentences provided for training.")
        raise HTTPException(status_code=400, detail="No sentences provided.")
    try:
        # Encode input sentences to embeddings
        embeddings = await model.encode(request.sentences)
        logger.debug(f"Encoded {len(request.sentences)} sentences for training")

        # Train the Faiss index with the embeddings
        await index.train(embeddings)
        logger.info("FaissIndex training completed.")
        return {"message": "Training complete"}
    except Exception as e:
        # Log any errors during training and return a 500 response
        logger.exception(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail="Training failed.")
    

@router.post("/embed", response_model=EmbeddingResponse)
async def embed(
    request: EmbeddingRequest, 
    model: SentenceEmbeddingModel = Depends(get_model),
    index: faiss.Index = Depends(get_faiss_index)
):
    """
    Endpoint to generate embeddings for input sentences and add them to the Faiss index.
    - Encodes input sentences using the SentenceEmbeddingModel.
    - Adds the resulting embeddings to the Faiss index.
    - Returns the embeddings in the response.
    """
    
    # Validate that the request contains sentences
    if not request.sentences:
        logger.warning("No sentences provided for embedding.")
        raise HTTPException(status_code=400, detail="No sentences provided.")
    try:
        # Encode input sentences to embeddings
        embeddings = await model.encode(request.sentences)
        logger.debug(f"Encoded {len(request.sentences)} sentences for embedding.")

        # Add embeddings to the Faiss index
        await index.add(embeddings)
        logger.info(f"Added {len(request.sentences)} embeddings to FaissIndex.")
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        # Log any errors during embedding and return a 500 response
        logger.exception(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed.")


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    model: SentenceEmbeddingModel = Depends(get_model),
    index: faiss.Index = Depends(get_faiss_index)
):
    """
    Endpoint to search the Faiss index for nearest neighbors to a query sentence.
    - Encodes the query sentence to an embedding.
    - Performs a search on the Faiss index to retrieve top-k closest items.
    - Returns distances and indices of the nearest neighbors.
    """
    
    # Validate that the request contains a query
    if not request.query:
        logger.warning("No query provided for search.")
        raise HTTPException(status_code=400, detail="No query provided.")
    try:
        # Encode the single query sentence to an embedding
        query_embedding = await model.encode([request.query])
        logger.debug("Encoded query for search.")

        # Search the Faiss index for top-k nearest neighbors
        distances, indices = await index.search(query_embedding, k=request.k)
        logger.info(f"Search completed: retrieved top {request.k} results.")
        return SearchResponse(distances=distances.tolist(), indices=indices.tolist())
    except Exception as e:
        # Log any errors during search and return a 500 response
        logger.exception(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Search failed.")
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch
import logging

class SentenceEmbeddingModel:
    """
    Wrapper class around HuggingFace SentenceTransformer model.
    Provides async-compatible methods for sentence encoding and cleanup.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize and load the sentence transformer model.

        Args:
            model_name (str): Hugging Face model name.
        """
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized SentenceTransformer model: {model_name}")

    async def encode(self, sentences: List[str]) -> List[List[float]]:
        """
        Encode a list of sentences into dense vector representations.

        Args:
            sentences (List[str]): List of input sentences.

        Returns:
            List[List[float]]: Encoded sentence embeddings as float32 numpy array.
        """
        embeddings = self.model.encode(sentences, convert_to_numpy=True).astype(np.float32)
        self.logger.debug(f"Encoding {len(sentences)} sentence(s).")
        return embeddings

    async def close(self):
        """
        Clean up resources used by the model.
        This includes freeing GPU memory (if used) and optionally clearing caches.
        """
        self.logger.info("Cleaning up sentence transformer model resources.")

        # Delete the model reference
        del self.model

        # Manually clear CUDA memory if running on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA memory cache cleared.")

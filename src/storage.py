import threading
import faiss
import numpy as np
from typing import Tuple
import logging

class FaissIndex:
    """
    Thread-safe wrapper around a FAISS IndexIVFFlat index for approximate nearest neighbor search.
    Supports asynchronous training, addition, searching, saving, and loading operations.
    """

    def __init__(self, dim: int, nlist: int = 100, metric: str = 'L2'):
        """
        Initializes the FAISS index with the given configuration.

        Args:
            dim (int): Dimensionality of the vectors.
            nlist (int): Number of clusters (inverted lists).
            metric (str): Similarity metric to use ('L2' or 'IP').
        """
        # Lock to ensure thread safety for write operations
        self.lock = threading.Lock()
        self.dim = dim
        self.nlist = nlist
        self.metric = faiss.METRIC_L2 if metric == 'L2' else faiss.METRIC_INNER_PRODUCT

        # Use a flat index as the quantizer
        self.quantizer = (
            faiss.IndexFlatL2(dim) if self.metric == faiss.METRIC_L2 
            else faiss.IndexFlatIP(dim)
        )

        # Create the IVF index
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, nlist, self.metric)
        self.is_trained = False

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized FAISS index (dim={dim}, nlist={nlist}, metric={metric}).")

    async def train(self, vectors: np.ndarray):
        """
        Trains the FAISS index on the provided vectors.

        Args:
            vectors (np.ndarray): Training vectors of shape (n_samples, dim).
        """
        with self.lock:
            if not self.index.is_trained:
                self.index.train(vectors)
                self.is_trained = True
            else:
                self.logger.debug("Index is already trained, skipping training.")

    async def add(self, vectors: np.ndarray):
        """
        Adds vectors to the FAISS index.

        Args:
            vectors (np.ndarray): Vectors to add of shape (n_samples, dim).

        Raises:
            ValueError: If index is not yet trained.
        """
        with self.lock:
            if not self.index.is_trained:
                self.logger.error("Attempted to add vectors before training the index.")
                raise ValueError("Index must be trained before adding.")
            self.index.add(vectors)
        self.logger.info(f"Added {vectors.shape[0]} vectors to FAISS index.")

    async def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches for the top-k nearest neighbors of each query vector.

        Args:
            queries (np.ndarray): Query vectors of shape (n_queries, dim).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of neighbors.

        Raises:
            ValueError: If index is not trained.
        """
        if not self.index.is_trained:
            self.logger.error("Search attempted on untrained index.")
            raise ValueError("Index must be trained before searching.")
        self.logger.debug(f"Performing FAISS search with k={k} for {queries.shape[0]} queries.")
        return self.index.search(queries, k)

    async def save(self, file_path: str):
        """
        Saves the current FAISS index to a file.

        Args:
            file_path (str): Path to the file where index should be saved.
        """
        with self.lock:
            faiss.write_index(self.index, file_path)
        self.logger.info(f"Saved FAISS index to '{file_path}'...")

    async def load(self, file_path: str):
        """
        Loads a FAISS index from a file.

        Args:
            file_path (str): Path to the saved index file.
        """
        with self.lock:
            self.index = faiss.read_index(file_path)
            self.is_trained = self.index.is_trained
        self.logger.info("Load complete. is_trained = %s", self.is_trained)

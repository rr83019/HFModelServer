from pydantic import BaseModel
from typing import List

class EmbeddingRequest(BaseModel):
    """
    Request schema for embedding sentences.

    Attributes:
        sentences (List[str]): A list of input sentences to be encoded.
    """
    sentences: List[str]

class EmbeddingResponse(BaseModel):
    """
    Response schema containing the embeddings for input sentences.

    Attributes:
        embeddings (List[List[float]]): A list of embedding vectors.
    """
    embeddings: List[List[float]]

class SearchRequest(BaseModel):
    """
    Request schema for performing a similarity search.

    Attributes:
        query (str): The input query sentence.
        k (int): The number of top results to return. Default is 5.
    """
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    """
    Response schema for search results.

    Attributes:
        distances (List[List[float]]): List of distance values for each query.
        indices (List[List[int]]): List of index positions for each query result.
    """
    distances: List[List[float]]
    indices: List[List[int]]

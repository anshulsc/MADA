# index/vector_store.py
import uuid
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import TextNode

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from config.config import (
    OPENAI_API_KEY,
    COLLECTION_NAME,
    VECTOR_SIZE,
    VECTOR_DISTANCE,
    QDRANT_HOST,
    QDRANT_PORT
)
from qdrant_client import QdrantClient, models
from utils.logger import logger

def initialize_vector_store():
    try:
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = VectorStoreIndex(
            embed_model=embed_model,
            store_nodes_override=True,
            index_struct=IndexDict(),
        )
        logger.info("Initialized VectorStoreIndex with OpenAIEmbedding.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise

def connect_qdrant():
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return qdrant_client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

def create_collection_if_not_exists(qdrant_client, collection_name):
    try:
        qdrant_client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        if "not found" in str(e).lower():
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance[VECTOR_DISTANCE]  # Ensure 'VECTOR_DISTANCE' is valid
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {VECTOR_SIZE} and distance {VECTOR_DISTANCE}.")
        else:
            logger.error(f"Error accessing collection info: {e}")
            raise

def add_documents_to_collection(documents, qdrant_client, collection_name, vector_store):
    try:
        nodes = [TextNode(text=doc.text) for doc in documents]
        embedded_nodes = vector_store._get_node_with_embedding(nodes)


  
        points = []
        for node in embedded_nodes:
            if hasattr(node, "embedding") and node.embedding is not None:
                embedding = node.embedding.tolist() if not isinstance(node.embedding, list) else node.embedding
                point = {
                    "id": str(uuid.uuid4()), 
                    "payload": {"document": node.text},
                    "vector": embedding,
                }
                points.append(point)

        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} documents to collection '{collection_name}'.")
        else:
            logger.warning("No points to upsert into Qdrant.")
    except Exception as e:
        logger.error(f"Failed to add documents to collection '{collection_name}': {e}")

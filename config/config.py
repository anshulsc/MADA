# config/settings.py
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# API Keys and Paths
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "MADA"

# Model Names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4"

# Qdrant Configuration
VECTOR_SIZE = 384  # Update based on your embedding model's output size
VECTOR_DISTANCE = "COSINE"  # Case-sensitive: 'Cosine', 'Euclidean', 'Dot'

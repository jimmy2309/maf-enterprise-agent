import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

def get_qdrant_client() -> QdrantClient:
    """Returns a connected Qdrant Client."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        raise e

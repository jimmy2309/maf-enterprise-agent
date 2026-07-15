import os
import sys

# Automatically add the project root to python path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams
from src.db.vector_store import get_qdrant_client

load_dotenv()

# The name of our collection in Qdrant
COLLECTION_NAME = "company_policies"

def setup_qdrant_collection(client):
    """Creates the collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        print(f"Creating new collection: {COLLECTION_NAME}")
        # FastEmbed default model 'BAAI/bge-small-en-v1.5' outputs 384 dimensions
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def ingest_documents():
    """Demonstrates Chunking Strategies and RAG Ingestion"""
    client = get_qdrant_client()
    setup_qdrant_collection(client)
    
    # We use FastEmbed because it runs locally, is free, and doesn't require OpenAI/Groq keys.
    embeddings = FastEmbedEmbeddings()

    # --- Our Dummy Data (Company Policy) ---
    policy_text = """
    EMPLOYEE HANDBOOK & COMPANY POLICY
    
    1. Background Check Policy
    All new employees must pass a comprehensive background check. If an applicant is found to have a criminal record involving theft, shoplifting, or fraud, they are immediately disqualified from employment. Minor traffic violations do not result in disqualification.
    
    2. Leave Policy
    Employees are entitled to 20 days of paid time off per year. Sick leave is unlimited but requires a medical certificate if taking more than 3 consecutive days.
    
    3. Code of Conduct
    Employees must maintain professional behavior at all times. Harassment of any kind will result in immediate termination.
    """

    print("\n--- Strategy 1: Recursive Character Chunking ---")
    # This is the industry standard fallback. It tries to split by paragraphs, then sentences, then words.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,  # Small chunk size for demonstration
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = recursive_splitter.create_documents([policy_text])
    print(f"Generated {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:2]): # Print first 2 to see
        print(f"Chunk {i+1}: {chunk.page_content}")

    print("\n--- Pushing to Qdrant ---")
    # Store the chunks in Qdrant Vector DB
    qdrant = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )
    
    qdrant.add_documents(chunks)
    print("✅ Documents successfully vectorized and stored in Qdrant!")

if __name__ == "__main__":
    ingest_documents()

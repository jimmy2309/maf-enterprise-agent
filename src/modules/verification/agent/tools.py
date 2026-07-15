from langchain_core.tools import tool
from src.db.vector_store import get_qdrant_client
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client.models import QueryResponse

@tool
def check_criminal_record(name: str) -> str:
    """Use this tool to check the criminal record of a given person."""
    # Mock API Logic
    if name.lower() == "john doe":
        return "CRIMINAL RECORD FOUND: Shoplifting conviction in 2019."
    return "CLEAN RECORD: No criminal history found."

@tool
def lookup_company_policy(query: str) -> str:
    """Use this tool to lookup the company's internal HR policies (e.g. background check rules, leave policy)."""
    # This tool performs an actual RAG query against our Qdrant Database!
    client = get_qdrant_client()
    embeddings = FastEmbedEmbeddings()
    
    # 1. Convert the user's query into a vector
    query_vector = embeddings.embed_query(query)
    
    # 2. Search Qdrant for the most relevant chunk
    search_result = client.search(
        collection_name="company_policies",
        query_vector=query_vector,
        limit=1  # Get top 1 result
    )
    
    if search_result:
        # Return the actual text chunk found in the DB
        return f"Company Policy Excerpt: {search_result[0].payload['page_content']}"
    
    return "No relevant policy found."

# Our list of tools
AVAILABLE_TOOLS = [check_criminal_record, lookup_company_policy]

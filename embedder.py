
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize local embedding model (1024-dim)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
model = SentenceTransformer(EMBEDDING_MODEL)



def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embeds chunks using a local Hugging Face model."""
    return model.encode(chunks, convert_to_numpy=True).tolist()



def embed_User_query(query: str) -> List[float]:
    """Embeds a user query using the local Hugging Face model."""
    return model.encode([query], convert_to_numpy=True)[0].tolist()
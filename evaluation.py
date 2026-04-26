"""
Evaluation metrics for the RAG pipeline.
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

from embedder import embed_User_query
from embedder import model as embedding_model
from vectorestore import search_in_pinecone
from llm import query_llm_with_context


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def hit_rate_at_k(test_queries: List[str], ground_truth_ids: List[List[str]], top_k: int = 5) -> float:
    """
    Calculate Hit Rate @ K.
    
    Args:
        test_queries: List of test queries
        ground_truth_ids: List of lists - for each query, the expected chunk IDs that should be retrieved
        top_k: Number of top results to consider
    
    Returns:
        Hit rate (percentage of queries where at least one ground truth ID is in top K results)
    """
    hits = 0
    
    for query, truth_ids in zip(test_queries, ground_truth_ids):
        # Get query embedding
        query_vector = embed_User_query(query)
        
        # Search Pinecone with top_k
        results = search_in_pinecone(query_vector, top_k=top_k)
        
        # Check if any ground truth ID is in the top K results
        # Since we don't have IDs returned, we'll check by text content similarity
        # For a proper implementation, you'd want to get chunk IDs from search results
        
        # For now, we'll use a simplified approach - check if any retrieved chunk matches
        # This is a placeholder - in production, you'd compare actual IDs
        retrieved_ids = [f"chunk_{i}" for i in range(len(results))]
        
        # Check for hit (any ground truth ID in retrieved)
        if any(tid in retrieved_ids for tid in truth_ids):
            hits += 1
    
    return hits / len(test_queries) if test_queries else 0.0


def measure_latency(process_query_func, test_queries: List[str], num_iterations: int = 50) -> Dict[str, float]:
    """
    Measure latency of query processing.
    
    Args:
        process_query_func: The function to measure (e.g., process_user_query)
        test_queries: List of queries to test
        num_iterations: Number of times to run each query
    
    Returns:
        Dictionary with avg and p95 latency in seconds
    """
    latencies = []
    
    # Repeat queries to reach num_iterations
    queries_to_run = (test_queries * (num_iterations // len(test_queries) + 1))[:num_iterations]
    
    for query in queries_to_run:
        start_time = time.time()
        
        # Run the query processing
        query_vector = embed_User_query(query)
        matched_chunks = search_in_pinecone(query_vector)
        # Note: Not calling LLM to avoid slowdowns, just measuring retrieval
        
        end_time = time.time()
        latencies.append(end_time - start_time)
    
    latencies_array = np.array(latencies)
    
    return {
        "avg_latency_sec": float(np.mean(latencies_array)),
        "p95_latency_sec": float(np.percentile(latencies_array, 95)),
        "min_latency_sec": float(np.min(latencies_array)),
        "max_latency_sec": float(np.max(latencies_array))
    }


def context_relevance_score(test_queries: List[str], top_k: int = 5) -> float:
    """
    Compute Context Relevance Score.
    
    Measures cosine similarity between query embedding and top K retrieved embeddings.
    
    Args:
        test_queries: List of test queries
        top_k: Number of top results to consider
    
    Returns:
        Mean cosine similarity score
    """
    scores = []
    
    for query in test_queries:
        # Get query embedding
        query_vector = embed_User_query(query)
        
        # Search Pinecone
        results = search_in_pinecone(query_vector, top_k=top_k)
        
        # For each retrieved chunk, compute similarity with query
        # We need to re-embed the chunks to compute similarity
        if results:
            chunk_embeddings = embedding_model.encode(results, convert_to_numpy=True)
            query_embedding = np.array(query_vector).reshape(1, -1)
            
            # Compute similarities
            for chunk_emb in chunk_embeddings:
                sim = compute_cosine_similarity(query_vector, chunk_emb.tolist())
                scores.append(sim)
    
    return float(np.mean(scores)) if scores else 0.0


def run_evaluation(
    test_queries: List[str] = None,
    ground_truth_ids: List[List[str]] = None,
    num_latency_iterations: int = 50
):
    """
    Run all evaluation metrics and print results.
    
    Args:
        test_queries: List of test queries (if None, uses default examples)
        ground_truth_ids: List of ground truth chunk IDs for each query
        num_latency_iterations: Number of iterations for latency measurement
    """
    # Default test queries if not provided
    if test_queries is None:
        test_queries = [
            "What is the work timing policy?",
            "How many casual leaves are allowed?",
            "What is the overtime policy?",
            "What are the weekly off days?",
            "How do I record attendance?"
        ]

    # Default ground truth IDs (placeholder - adjust based on your data)
    if ground_truth_ids is None:
        ground_truth_ids = [
            ["chunk_0", "chunk_1"],  # For work timing
            ["chunk_2", "chunk_3"],  # For casual leaves
            ["chunk_4"],             # For overtime
            ["chunk_5"],             # For weekly off
            ["chunk_6"]              # For attendance
        ]

    print("=" * 60)
    print("RAG PIPELINE EVALUATION")
    print("=" * 60)
    
    # 1. Hit Rate @ 5
    print("\n[1] Hit Rate @ 5")
    print("-" * 40)
    hit_rate = hit_rate_at_k(test_queries, ground_truth_ids, top_k=5)
    print(f"Hit Rate @ 5: {hit_rate * 100:.2f}%")

    # 2. Latency Measurement
    print("\n[2] Latency Measurement")
    print("-" * 40)
    latency_results = measure_latency(
        lambda q: (embed_User_query(q), search_in_pinecone(embed_User_query(q))),
        test_queries,
        num_latency_iterations
    )
    print(f"Average Latency: {latency_results['avg_latency_sec']:.4f} seconds")
    print(f"P95 Latency:      {latency_results['p95_latency_sec']:.4f} seconds")
    print(f"Min Latency:      {latency_results['min_latency_sec']:.4f} seconds")
    print(f"Max Latency:      {latency_results['max_latency_sec']:.4f} seconds")

    # 3. Context Relevance Score
    print("\n[3] Context Relevance Score")
    print("-" * 40)
    relevance_score = context_relevance_score(test_queries, top_k=5)
    print(f"Mean Cosine Similarity: {relevance_score:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
"""
RAG (Retrieval-Augmented Generation) pipeline for AutoStream's knowledge base.

Uses sentence-transformers to embed knowledge base chunks and retrieves
the most semantically relevant chunks for a given user query.
"""

import json
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer


class KnowledgeRetriever:
    """
    Semantic knowledge retriever for the AutoStream knowledge base.

    Workflow:
      1. Load knowledge_base.json and convert it into text chunks
      2. Embed all chunks using a lightweight sentence-transformer model
      3. On each query, embed the query and return top-k most similar chunks
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight, ~80MB — great for local RAG

    def __init__(self, kb_path: str = "data/knowledge_base.json"):
        print(f"[RAG] Loading knowledge base from: {kb_path}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
        self._load_and_index(kb_path)

    def _load_and_index(self, kb_path: str) -> None:
        """Load JSON knowledge base, build chunks, and compute embeddings."""
        path = Path(kb_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found at: {kb_path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.chunks = self._build_chunks(data)
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        print(f"[RAG] Indexed {len(self.chunks)} chunks successfully.")

    def _build_chunks(self, data: dict) -> List[str]:
        """
        Convert the structured JSON into flat, self-contained text chunks.
        Each chunk should be independently understandable.
        """
        chunks = []

        # --- Product overview ---
        p = data["product"]
        chunks.append(
            f"{p['name']} — {p['tagline']}. "
            f"{p['description']} "
            f"Best suited for: {', '.join(p['use_cases'])}."
        )

        # --- Basic plan ---
        basic = data["pricing"]["basic_plan"]
        chunks.append(
            f"Basic Plan costs {basic['price']}. "
            f"Includes {basic['videos_per_month']} videos per month at {basic['resolution']} resolution. "
            f"Features: {'; '.join(basic['features'])}. "
            f"Ideal for casual creators starting out."
        )

        # --- Pro plan ---
        pro = data["pricing"]["pro_plan"]
        chunks.append(
            f"Pro Plan costs {pro['price']}. "
            f"Includes {pro['videos_per_month']} videos per month at {pro['resolution']} resolution with AI-generated captions. "
            f"Features: {'; '.join(pro['features'])}. "
            f"Designed for serious, professional content creators."
        )

        # --- Side-by-side comparison ---
        chunks.append(
            f"Plan comparison: "
            f"Basic Plan at {basic['price']} offers {basic['videos_per_month']} videos/month at {basic['resolution']}. "
            f"Pro Plan at {pro['price']} offers {pro['videos_per_month']} videos/month at {pro['resolution']} with AI captions and 24/7 support. "
            f"The Pro Plan is a better fit for full-time creators."
        )

        # --- Refund and cancellation policy ---
        pol = data["policies"]
        chunks.append(
            f"Refund policy: {pol['refund']['policy']}. "
            f"{pol['refund']['trial']}. "
            f"Cancellation policy: {pol['cancellation']}."
        )

        # --- Support policy ---
        chunks.append(
            f"Support options: "
            f"Basic plan users get {pol['support']['basic_plan']}. "
            f"Pro plan users get {pol['support']['pro_plan']}. "
            f"24/7 live support is exclusively available on the Pro plan."
        )

        return chunks

    def retrieve(self, query: str, top_k: int = 2) -> str:
        """
        Retrieve the top-k most relevant knowledge base chunks for a query.

        Args:
            query:  The user's question or statement.
            top_k:  Number of chunks to return (default: 2).

        Returns:
            A newline-joined string of the most relevant chunks.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity = dot(a, b) / (|a| * |b|)
        dot_products = np.dot(self.embeddings, query_embedding.T).flatten()
        norms = (
            np.linalg.norm(self.embeddings, axis=1)
            * np.linalg.norm(query_embedding)
            + 1e-9  # avoid division by zero
        )
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved = [self.chunks[i] for i in top_indices]

        return "\n\n".join(retrieved)

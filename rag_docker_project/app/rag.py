"""Core RAG pipeline: load -> clean -> chunk -> embed -> retrieve -> generate."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the numbered "
    "context passages provided. Cite passages like [1] or [2]. If the context does "
    "not contain the answer, reply exactly: \"I don't know based on the provided documents.\""
)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str) -> dict:
    docs = {}
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs[path.name] = clean_text(path.read_text(encoding="utf-8"))
    return docs


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list:
    """Split text into word-based chunks with overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


class RAGPipeline:
    def __init__(self, data_dir=None, embed_model=None, llm_model=None,
                 top_k=None, chunk_size=None, chunk_overlap=None):
        from sentence_transformers import SentenceTransformer

        self.data_dir = data_dir or os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
        self.embed_model_name = embed_model or os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.top_k = int(top_k or os.getenv("TOP_K", 4))
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", 120))
        self.chunk_overlap = int(chunk_overlap or os.getenv("CHUNK_OVERLAP", 30))

        self.model = SentenceTransformer(self.embed_model_name)
        self.chunks = []
        self.embeddings = None
        self.build_index()

    def build_index(self):
        docs = load_documents(self.data_dir)
        if not docs:
            raise RuntimeError(f"No .txt or .md documents found in {self.data_dir}")
        self.chunks = []
        for name, text in docs.items():
            for i, piece in enumerate(chunk_text(text, self.chunk_size, self.chunk_overlap)):
                self.chunks.append(Chunk(text=piece, source=name, chunk_id=i))
        self.embeddings = self.model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k=None):
        """Return the top-k (Chunk, cosine_score) pairs."""
        k = int(k or self.top_k)
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def generate(self, query: str, retrieved) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_api_key_here":
            return ("(No OPENAI_API_KEY set - showing retrieved passages only. "
                    "Add a key to .env to generate answers.)")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = "\n\n".join(
            f"[{i}] (source: {c.source})\n{c.text}"
            for i, (c, _) in enumerate(retrieved, 1)
        )
        resp = client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return resp.choices[0].message.content.strip()

    def ask(self, query: str, k=None) -> dict:
        retrieved = self.retrieve(query, k)
        return {"answer": self.generate(query, retrieved), "sources": retrieved}

"""
Ingestion: parse -> chunk -> embed -> upsert into ChromaDB.
"""
import argparse
from collections import Counter
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

import chunkers

DATA = Path("data")
DB_PATH = "chroma_db"
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["structure", "fixed", "both"], default="both")
    ap.add_argument("--model", choices=list(MODELS), default="minilm")
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--show-columns", action="store_true")
    args = ap.parse_args()

    if args.show_columns:
        df = pd.read_csv(DATA / "cards.csv")
        print(df.columns.tolist())
        print(df.head(2).T)
        return

    model = SentenceTransformer(MODELS[args.model])
    client = chromadb.PersistentClient(path=DB_PATH)
    strategies = ["structure", "fixed"] if args.strategy == "both" else [args.strategy]

    for strategy in strategies:
        if strategy == "structure":
            chunks = chunkers.structure_chunks(DATA)
        else:
            chunks = chunkers.fixed_chunks(DATA, model.tokenizer, args.chunk_size, args.overlap)

        name = f"tarot_{strategy}_{args.model}"
        try:
            client.delete_collection(name)  # rebuild from scratch on every run
        except Exception:
            pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

        for i in range(0, len(chunks), 64):
            batch = chunks[i:i + 64]
            embeddings = model.encode([c["text"] for c in batch], normalize_embeddings=True).tolist()
            col.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                embeddings=embeddings,
            )

        too_long = sum(len(model.tokenizer(c["text"])["input_ids"]) > model.max_seq_length for c in chunks)
        print(f"{name}: {len(chunks)} chunks {dict(Counter(c['metadata']['type'] for c in chunks))}, "
              f"{too_long} longer than the model's {model.max_seq_length}-token limit (truncated when embedded)")


if __name__ == "__main__":
    main()
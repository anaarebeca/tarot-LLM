"""
Quick retrieval check.
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import MODELS, DB_PATH

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--strategy", choices=["structure", "fixed"], default="structure")
ap.add_argument("--model", choices=list(MODELS), default="minilm")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--card")
ap.add_argument("--orientation")
args = ap.parse_args()

conds = []
if args.card:
    conds.append({"card": args.card})
if args.orientation:
    conds.append({"orientation": args.orientation})
where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

col = chromadb.PersistentClient(path=DB_PATH).get_collection(f"tarot_{args.strategy}_{args.model}")
emb = SentenceTransformer(MODELS[args.model]).encode([args.query], normalize_embeddings=True).tolist()
res = col.query(query_embeddings=emb, n_results=args.k, where=where)

for id_, dist, doc in zip(res["ids"][0], res["distances"][0], res["documents"][0]):
    print(f"{id_}  (distance {dist:.3f})\n  {doc[:150]!r}\n")
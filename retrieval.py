from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import MODELS, DB_PATH
from chunkers import slug


@lru_cache(maxsize=None)
def get_model(name):
    return SentenceTransformer(MODELS[name])


@lru_cache(maxsize=None)
def get_collection(strategy, model_name):
    return chromadb.PersistentClient(path=DB_PATH).get_collection(f"tarot_{strategy}_{model_name}")


def naive(strategy, model_name, query, k=5, where=None):
    #semantic top-k - returns a list of (id, text, metadata)
    emb = get_model(model_name).encode([query], normalize_embeddings=True).tolist()
    res = get_collection(strategy, model_name).query(query_embeddings=emb, n_results=k, where=where)
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))


def metadata_filtered(strategy, model_name, query, k=5, card=None, orientation=None, type=None):
    #exact lookup for a known card, semantic search restricted to a chunk type ohterwise
    if card and orientation:
        res = get_collection(strategy, model_name).get(ids=[f"card::{slug(card)}::{orientation}"])
        return list(zip(res["ids"], res["documents"], res["metadatas"]))
    return naive(strategy, model_name, query, k, where={"type": type} if type else None)
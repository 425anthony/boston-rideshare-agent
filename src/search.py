search_code = '''"""
search.py - Search method for Boston Rideshare Agent
Implements TF-IDF based similarity search for historical trips.
"""

import re
import math
from collections import defaultdict
from typing import List, Dict, Any


def tokenize(text: str) -> List[str]:
    """Tokenize text into words using regex."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for each word."""
    counts = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    length = max(1, len(tokens))
    return {token: counts[token] / length for token in counts}


def compute_df(doc_tokens: List[List[str]]) -> Dict[str, int]:
    """Compute document frequency across corpus."""
    df = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    return df


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF vector for a document."""
    tf = compute_tf(tokens)
    return {word: tf[word] * idf.get(word, 0.0) for word in tf}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (na * nb + 1e-12)


def search_corpus(query: str, corpus: List[Dict], doc_vecs: List[Dict], 
                 idf: Dict[str, float], k: int = 3) -> List[Dict[str, Any]]:
    """Search corpus for most similar documents."""
    qvec = tfidf_vector(tokenize(query), idf)
    scored = [(i, cosine(qvec, v)) for i, v in enumerate(doc_vecs)]
    scored.sort(reverse=True, key=lambda x: x[1])
    
    results = []
    for score, idx in scored[:k]:
        d = corpus[idx].copy()
        d["score"] = float(score)
        results.append(d)
    return results


def tool_search(query: str, corpus: List[Dict], doc_vecs: List[Dict],
               idf: Dict[str, float], k: int = 3) -> Dict[str, Any]:
    """Tool wrapper for agent integration."""
    hits = search_corpus(query, corpus, doc_vecs, idf, k)
    return {
        "tool": "search",
        "query": query,
        "results": [
            {
                "id": h["id"],
                "title": h["title"],
                "snippet": h["text"][:240] + ("..." if len(h["text"]) > 240 else ""),
                "price": h["price"],
                "cab_type": h["cab_type"],
                "surge": h["surge_multiplier"],
                "distance": h["distance"],
                "score": h["score"]
            }
            for h in hits
        ]
    }
'''
# Retrieval functions (Joel)


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index over the corpus.
    corpus: [{"doc_id": ..., "text": ..., "pmid": ...}]
    """

def bm25_retrieve(query: str, 
                  index: BM25Okapi, 
                  corpus: list[dict], 
                  top_k: int = 10) -> list[dict]:
    """
    Retrieve top-K documents using BM25 sparse matching.
    Returns: [{"doc_id", "text", "pmid", "score"}]
    """

def fetch_pubmed_abstract(pmid: str) -> dict:
    """
    Fetch abstract from PubMed via E-utilities API given a PMID.
    Returns: {"pmid", "title", "abstract"}
    """

def build_corpus_from_bioasq(questions: list[dict]) -> list[dict]:
    """
    Extract all unique documents from BioASQ questions 
    and fetch their abstracts to build the retrieval corpus.
    """

def build_dense_index(corpus: list[dict], 
                      model_name: str = "BAAI/bge-m3") -> tuple:
    """
    Encode corpus with a biomedical embedding model.
    Returns: (embeddings_matrix, encoder_model)
    """

def dense_retrieve(query: str, 
                   embeddings, 
                   encoder, 
                   corpus: list[dict], 
                   top_k: int = 10) -> list[dict]:
    """
    Retrieve top-K documents via cosine similarity over dense embeddings.
    Returns: [{"doc_id", "text", "pmid", "score"}]
    """

def reciprocal_rank_fusion(bm25_results: list[dict], 
                           dense_results: list[dict], 
                           k: int = 60) -> list[dict]:
    """
    Merge sparse and dense results using RRF scoring.
    Returns: reranked list of [{"doc_id", "text", "pmid", "rrf_score"}]
    """

def rerank_with_crossencoder(query: str, 
                              candidates: list[dict], 
                              model_name: str, 
                              top_k: int = 5) -> list[dict]:
    """
    Apply cross-encoder reranking on fused candidates.
    Returns: final top-K [{"doc_id", "text", "pmid", "rerank_score"}]
    """
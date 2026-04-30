"""
mcp_tools/search_lambda/handler.py
====================================
Stage 1 + Stage 2 + Stage 4a of the Advanced RAG Pipeline.

MCP tool schema (from deploy.py gateway registration):
  Input:  { "query": str, "top_k": int }
  Output: str  — newline-separated chunks with reranker scores,
                 or INSUFFICIENT_CONTEXT sentinel

Pipeline:
  Stage 1 — Pinecone dense vector search: top_k=100 candidates
  Stage 2 — Cohere Rerank: cross-encoder scores every (query, chunk)
             jointly. Cuts to top 20 by relevance score.
  Stage 4a — Confidence gate: if top reranker score < RERANK_THRESHOLD,
              return INSUFFICIENT_CONTEXT so the research agent returns
              "I don't have enough information" instead of hallucinating.

Why top_k=100 not the caller's top_k?
  The caller (research agent LLM) was trained to pass top_k=10.
  Stage 1 must retrieve broadly (100) to give Stage 2 enough candidates
  to find the truly relevant chunks. The final output is still ≤20 chunks.

Env / SSM:
  SSM_PREFIX                    /vs-agentcore-multiagent/prod
  {SSM_PREFIX}/pinecone         {"api_key": "..."}
  {SSM_PREFIX}/cohere           {"api_key": "..."}
  {SSM_PREFIX}/pinecone/clinical_trials_index
"""

import json
import logging
import os

import boto3

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────
STAGE1_TOP_K       = 100     # broad recall — must be high to feed reranker
STAGE2_TOP_N       = 20      # reranker keeps top N after scoring
RERANK_THRESHOLD   = 0.3     # confidence gate: below this = no evidence
SSM_PREFIX         = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
REGION             = os.environ.get("AWS_REGION", "us-east-1")

# Sentinel returned when no chunk clears the confidence threshold.
# The research agent prompt recognises this and returns a grounded refusal.
INSUFFICIENT_CONTEXT = (
    "INSUFFICIENT_CONTEXT: No chunks in the knowledge base scored above the "
    "relevance threshold for this query. Do not hallucinate an answer. "
    "Return: 'I don't have enough information in the knowledge base to answer this question.'"
)

# ── Lazy-loaded clients ────────────────────────────────────────────────────
_secrets: dict = {}
_pinecone_index = None
_cohere_client  = None


def _load_secrets() -> dict:
    global _secrets
    if _secrets:
        return _secrets
    sm = boto3.client("secretsmanager", region_name=REGION)
    pinecone_secret = json.loads(
        sm.get_secret_value(SecretId=f"{SSM_PREFIX}/pinecone")["SecretString"]
    )
    cohere_secret = json.loads(
        sm.get_secret_value(SecretId=f"{SSM_PREFIX}/cohere")["SecretString"]
    )
    _secrets = {
        "pinecone_api_key": pinecone_secret["api_key"],
        "cohere_api_key":   cohere_secret["api_key"],
    }
    log.info("[search] Secrets loaded")
    return _secrets


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index:
        return _pinecone_index
    from pinecone import Pinecone
    secrets    = _load_secrets()
    ssm        = boto3.client("ssm", region_name=REGION)
    index_name = ssm.get_parameter(
        Name=f"{SSM_PREFIX}/pinecone/clinical_trials_index"
    )["Parameter"]["Value"]
    pc             = Pinecone(api_key=secrets["pinecone_api_key"])
    _pinecone_index = pc.Index(index_name)
    log.info(f"[search] Pinecone index ready: {index_name}")
    return _pinecone_index


def _get_cohere_client():
    global _cohere_client
    if _cohere_client:
        return _cohere_client
    import cohere
    secrets        = _load_secrets()
    _cohere_client = cohere.Client(api_key=secrets["cohere_api_key"])
    log.info("[search] Cohere client ready")
    return _cohere_client


# ── Pipeline stages ────────────────────────────────────────────────────────

def _stage1_broad_recall(query: str) -> list[dict]:
    """
    Stage 1: Pinecone dense vector search.
    Returns top STAGE1_TOP_K chunks as dicts with {id, text, metadata}.
    Optimised for recall — missing the right document here is fatal.
    """
    from openai import OpenAI
    import json as _json

    # Load OpenAI key for embeddings
    if not os.environ.get("OPENAI_API_KEY"):
        sm     = boto3.client("secretsmanager", region_name=REGION)
        secret = _json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"])
        os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")

    oai_client = OpenAI()
    embedding  = oai_client.embeddings.create(
        model = "text-embedding-3-large",
        input = query,
    ).data[0].embedding

    index   = _get_pinecone_index()
    results = index.query(
        vector          = embedding,
        top_k           = STAGE1_TOP_K,
        include_metadata = True,
    )

    chunks = []
    for match in results.get("matches", []):
        text = (
            match.get("metadata", {}).get("text")
            or match.get("metadata", {}).get("chunk")
            or match.get("metadata", {}).get("content")
            or ""
        )
        if text:
            chunks.append({
                "id":       match.get("id", ""),
                "text":     text,
                "metadata": match.get("metadata", {}),
                "score":    match.get("score", 0.0),
            })

    log.info(f"[search] Stage 1: {len(chunks)} chunks from Pinecone")
    return chunks


def _stage2_rerank(query: str, chunks: list[dict]) -> tuple[list[dict], float]:
    """
    Stage 2: Cohere Rerank cross-encoder.
    Scores every (query, chunk) pair jointly — no compression, full attention.
    Returns (top_n_chunks_sorted_by_relevance, top_score).

    The cross-encoder reads the full text of both query and chunk together,
    so it correctly handles:
      - Negation:    "no adverse events" != "adverse events"
      - Synonyms:    "elderly" == "senior subjects aged 65+"
      - Specificity: subgroup results > overall results for specific query
      - Contradiction: "excluded" ranks low even with keyword overlap
    """
    co     = _get_cohere_client()
    docs   = [c["text"] for c in chunks]

    result = co.rerank(
        model     = "rerank-english-v3.0",
        query     = query,
        documents = docs,
        top_n     = STAGE2_TOP_N,
    )

    reranked = []
    for hit in result.results:
        chunk = chunks[hit.index].copy()
        chunk["rerank_score"] = hit.relevance_score
        reranked.append(chunk)

    top_score = reranked[0]["rerank_score"] if reranked else 0.0
    log.info(
        f"[search] Stage 2: {len(reranked)} chunks after rerank  "
        f"top_score={top_score:.3f}"
    )
    return reranked, top_score


def _format_chunk(chunk: dict) -> str:
    """
    Format a chunk for return to the research agent LLM.
    Includes the NCT ID (if present) so Stage 4b citation grounding can work.
    """
    meta   = chunk.get("metadata", {})
    nct_id = meta.get("nct_id") or meta.get("trial_id") or meta.get("source") or ""
    score  = chunk.get("rerank_score", chunk.get("score", 0.0))
    text   = chunk.get("text", "").strip()

    header = f"[Relevance: {score:.2f}"
    if nct_id:
        header += f" | Source: {nct_id}"
    header += "]"

    return f"{header}\n{text}"


# ── Lambda entrypoint ──────────────────────────────────────────────────────

def handler(event: dict, context) -> str:
    """
    MCP Lambda handler — called by the Bedrock MCP Gateway.

    Input (from LangChain tool call):
        event["query"]  : str — the research query
        event["top_k"]  : int — ignored; we always use STAGE1_TOP_K for recall

    Output:
        str — formatted chunks for the research agent LLM,
              or INSUFFICIENT_CONTEXT sentinel
    """
    query  = event.get("query", "").strip()
    if not query:
        return "Error: empty query"

    log.info(f"[search] Query: {query[:100]}")

    try:
        # Stage 1 — Broad recall
        chunks = _stage1_broad_recall(query)
        if not chunks:
            log.warning("[search] Stage 1 returned 0 chunks")
            return INSUFFICIENT_CONTEXT

        # Stage 2 — Rerank
        reranked, top_score = _stage2_rerank(query, chunks)
        if not reranked:
            return INSUFFICIENT_CONTEXT

        # Stage 4a — Confidence gate
        if top_score < RERANK_THRESHOLD:
            log.warning(
                f"[search] Confidence gate triggered: top_score={top_score:.3f} "
                f"< threshold={RERANK_THRESHOLD}"
            )
            return INSUFFICIENT_CONTEXT

        # Format and return top chunks
        formatted = "\n\n---\n\n".join(_format_chunk(c) for c in reranked)
        log.info(f"[search] Returning {len(reranked)} chunks  top_score={top_score:.3f}")
        return formatted

    except Exception as exc:
        log.exception(f"[search] Pipeline error: {exc}")
        return f"Error retrieving documents: {exc}"
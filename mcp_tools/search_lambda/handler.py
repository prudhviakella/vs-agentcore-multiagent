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
RERANK_THRESHOLD   = 0.15    # confidence gate: corpus uses AI-generated summaries
                             # which rerankers score lower than raw protocol text
SSM_PREFIX         = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
REGION             = os.environ.get("AWS_REGION", "us-east-1")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "clinical-trials")

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
    # Index dimension is 1536.
    # Try text-embedding-3-small first (native 1536, likely matches ingestion model).
    # If ingestion used ada-002, cosine similarity still works well across both models
    # since they share similar semantic space for clinical text.
    embedding  = oai_client.embeddings.create(
        model = "text-embedding-3-small",
        input = query,
    ).data[0].embedding
    log.info(f"[search] Embedding dim={len(embedding)}")

    index   = _get_pinecone_index()
    results = index.query(
        vector           = embedding,
        top_k            = STAGE1_TOP_K,
        namespace        = PINECONE_NAMESPACE,
        include_metadata = True,
    )
    log.info(f"[search] Stage 1: Pinecone query namespace={PINECONE_NAMESPACE!r}  matches={len(results.get('matches', []))}")

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
    Includes the NCT ID so Stage 4b citation grounding can work.

    The Pinecone metadata has no nct_id field — NCT IDs are embedded
    directly in the chunk text (e.g. "NCT04470427") or in S3 paths
    (e.g. doc_NCT04470427_Pfizer_Vaccine_...). Extract via regex.
    """
    import re as _re
    meta   = chunk.get("metadata", {})
    score  = chunk.get("rerank_score", chunk.get("score", 0.0))
    text   = chunk.get("text", "").strip()

    # Try metadata fields first (future-proof)
    nct_id = meta.get("nct_id") or meta.get("trial_id") or meta.get("source") or ""

    # Fall back to regex extraction from text
    if not nct_id:
        _nct_match = _re.search(r'NCT\d{8}', text)
        if _nct_match:
            nct_id = _nct_match.group(0)

    # Fall back to S3 path in text
    if not nct_id:
        _s3_match = _re.search(r'doc_(NCT\d{8})', text)
        if _s3_match:
            nct_id = _s3_match.group(1)

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
        str — JSON envelope containing:
            chunks            : formatted chunk text for the LLM
            rag_metrics       : structured observability fields for the tracer
                rerank_top_score      : float  — highest relevance score from Stage 2
                rerank_threshold      : float  — threshold used (RERANK_THRESHOLD)
                threshold_triggered   : bool   — true if confidence gate fired
                chunks_stage1         : int    — candidates from Pinecone
                chunks_stage2         : int    — chunks after reranking
            or INSUFFICIENT_CONTEXT sentinel if confidence gate triggered
    """
    query  = event.get("query", "").strip()
    if not query:
        return "Error: empty query"

    log.info(f"[search] Query: {query[:100]}")

    try:
        # Stage 1 — Broad recall
        chunks = _stage1_broad_recall(query)
        chunks_stage1 = len(chunks)
        if not chunks:
            log.warning("[search] Stage 1 returned 0 chunks")
            return json.dumps({
                "chunks": INSUFFICIENT_CONTEXT,
                "rag_metrics": {
                    "rerank_top_score":    0.0,
                    "rerank_threshold":    RERANK_THRESHOLD,
                    "threshold_triggered": True,
                    "chunks_stage1":       0,
                    "chunks_stage2":       0,
                }
            })

        # Stage 2 — Rerank
        reranked, top_score = _stage2_rerank(query, chunks)
        chunks_stage2 = len(reranked)

        # Stage 4a — Confidence gate
        threshold_triggered = (not reranked) or (top_score < RERANK_THRESHOLD)
        if threshold_triggered:
            log.warning(
                f"[search] Confidence gate triggered: top_score={top_score:.3f} "
                f"< threshold={RERANK_THRESHOLD}"
            )
            # Still capture top chunk IDs even when gated — tells learning pipeline
            # whether wrong docs are ranking near threshold vs no docs at all
            _top_ids    = [c.get("id", "") for c in reranked[:5]]
            _top_scores = [round(c.get("rerank_score", 0.0), 4) for c in reranked[:5]]
            _rag = json.dumps({
                "rerank_top_score":    round(top_score, 4),
                "rerank_threshold":    RERANK_THRESHOLD,
                "threshold_triggered": True,
                "chunks_stage1":       chunks_stage1,
                "chunks_stage2":       chunks_stage2,
                "top_chunk_ids":       _top_ids,
                "top_chunk_scores":    _top_scores,
            })
            return f"RAG_METRICS:{_rag}\n\n{INSUFFICIENT_CONTEXT}"

        # Format top chunks for LLM — plain text, no JSON wrapper
        # The LLM must be able to pass this directly to the summariser without parsing.
        # Limit to top 10 chunks to stay within LLM context window for reliable passthrough.
        TOP_N_LEARNING = 5
        top_chunk_ids    = [c.get("id", "") for c in reranked[:TOP_N_LEARNING]]
        top_chunk_scores = [round(c.get("rerank_score", 0.0), 4) for c in reranked[:TOP_N_LEARNING]]
        display_chunks   = reranked[:10]  # LLM sees top 10

        formatted = "\n\n---\n\n".join(_format_chunk(c) for c in display_chunks)

        # Embed rag_metrics as a machine-readable header that research_app.py
        # extracts via on_tool_end, while the LLM treats the whole response as text.
        rag_metrics = json.dumps({
            "rerank_top_score":    round(top_score, 4),
            "rerank_threshold":    RERANK_THRESHOLD,
            "threshold_triggered": False,
            "chunks_stage1":       chunks_stage1,
            "chunks_stage2":       chunks_stage2,
            "top_chunk_ids":       top_chunk_ids,
            "top_chunk_scores":    top_chunk_scores,
        })

        log.info(
            f"[search] Done  chunks_s1={chunks_stage1}  chunks_s2={chunks_stage2}  "
            f"top_score={top_score:.3f}  returning {len(display_chunks)} chunks to LLM"
        )

        # Return format: RAG_METRICS header (parsed by research_app) + plain chunk text
        return f"RAG_METRICS:{rag_metrics}\n\n{formatted}"

    except Exception as exc:
        log.exception(f"[search] Pipeline error: {exc}")
        return json.dumps({
            "chunks": f"Error retrieving documents: {exc}",
            "rag_metrics": {
                "rerank_top_score":    0.0,
                "rerank_threshold":    RERANK_THRESHOLD,
                "threshold_triggered": False,
                "chunks_stage1":       0,
                "chunks_stage2":       0,
                "error":               str(exc),
            }
        })
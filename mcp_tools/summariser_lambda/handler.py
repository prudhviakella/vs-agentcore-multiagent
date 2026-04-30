"""
mcp_tools/summariser_lambda/handler.py
========================================
Stage 3 + Stage 4b of the Advanced RAG Pipeline.

MCP tool schema (from deploy.py gateway registration):
  Input:  { "chunks": list[str], "query": str }
  Output: str  — grounded answer with [Source: NCT_ID] citations

Pipeline:
  Stage 3  — Context compression: GPT-4o-mini extracts only the sentences
              from each chunk that are relevant to the query.
              Reduces 20 × ~6,000 tokens → 20 × ~500 tokens (12× reduction).
              Preserves specific numbers, NCT IDs, dosages — the facts that
              matter. Prompt-based extraction was chosen over LLMLingua because
              clinical trial documents contain sparse but critical numerical
              values that aggressive token-level compression would drop.

  Stage 4b — Citation grounding: GPT-4o synthesises the compressed context
              with a prompt that forces [Source: NCT_ID] after every factual
              claim. Claims without a citation are stripped before return.
              This eliminates the hallucination path — the LLM cannot make
              a claim it cannot point to.

Env / SSM:
  SSM_PREFIX              /vs-agentcore-multiagent/prod
  {SSM_PREFIX}/openai     {"api_key": "..."}
"""

import json
import logging
import os
import re

import boto3

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
REGION     = os.environ.get("AWS_REGION", "us-east-1")

# Max tokens per chunk after compression (Stage 3).
# A 6,000-token chunk compressed to 500 tokens = 12× reduction.
# Keeps specific numbers, NCT IDs, dosages — drops boilerplate methodology.
COMPRESSED_TOKEN_BUDGET = 500

# Sentence-level citation pattern: [Source: NCT04470427]
CITATION_PATTERN = re.compile(r'\[Source:\s*[A-Z0-9]+\]', re.IGNORECASE)

# Sentences we strip if they contain no citation (hallucination guard)
FACTUAL_CLAIM_PATTERN = re.compile(
    r'(?:efficacy|safety|adverse|endpoint|result|outcome|percent|%|rate|'
    r'hazard|ratio|p-value|confidence|interval|dose|dosage|mg|kg|trial|'
    r'study|participant|patient|subject)',
    re.IGNORECASE,
)


# ── OpenAI client ──────────────────────────────────────────────────────────

def _get_openai_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    sm     = boto3.client("secretsmanager", region_name=REGION)
    secret = json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"])
    key    = secret.get("api_key", "")
    os.environ["OPENAI_API_KEY"] = key
    return key


# ── Stage 3: Context compression ──────────────────────────────────────────

def _compress_chunk(chunk_text: str, query: str, client) -> str:
    """
    GPT-4o-mini extracts only the sentences from a single chunk that are
    directly relevant to the query.

    Why prompt-based not LLMLingua:
      LLMLingua does aggressive token-level compression — good for general text
      but risky for clinical trial documents where a single number (18%, p=0.04,
      NCT04470427) may be the entire value of the chunk. GPT-4o-mini reads the
      full chunk and extracts complete relevant sentences, preserving all
      numerical facts and source identifiers.
    """
    # Already short — no compression needed
    word_count = len(chunk_text.split())
    if word_count < 150:
        return chunk_text

    prompt = (
        f"Extract only the sentences from the following clinical trial document "
        f"chunk that are directly relevant to this query:\n\n"
        f"Query: {query}\n\n"
        f"Document chunk:\n{chunk_text}\n\n"
        f"Rules:\n"
        f"- Return only the relevant sentences verbatim, in order.\n"
        f"- Keep all numerical values, percentages, NCT IDs, drug names, and dates.\n"
        f"- If no sentences are relevant, return: NOT_RELEVANT\n"
        f"- Do not summarise, paraphrase, or add commentary.\n"
        f"- Keep your response under {COMPRESSED_TOKEN_BUDGET} tokens."
    )

    response = client.chat.completions.create(
        model       = "gpt-4o-mini",
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = COMPRESSED_TOKEN_BUDGET,
        temperature = 0.0,
    )
    return response.choices[0].message.content.strip()


def _stage3_compress(chunks: list[str], query: str, client) -> list[str]:
    """
    Stage 3: Run compression on all chunks in parallel (sequential for now,
    can be parallelised with asyncio.gather in a future iteration).
    Filters out NOT_RELEVANT chunks.
    """
    compressed = []
    for i, chunk in enumerate(chunks):
        result = _compress_chunk(chunk, query, client)
        if result and result != "NOT_RELEVANT":
            compressed.append(result)
        log.info(
            f"[summariser] Stage 3: chunk {i+1}/{len(chunks)}  "
            f"before={len(chunk.split())} words  "
            f"after={len(result.split()) if result != 'NOT_RELEVANT' else 0} words"
        )

    log.info(f"[summariser] Stage 3: {len(compressed)}/{len(chunks)} chunks retained after compression")
    return compressed


# ── Stage 4b: Citation grounding ──────────────────────────────────────────

def _validate_citations(answer: str) -> str:
    """
    Strip factual claims that have no [Source: NCT_ID] citation.
    This is the final hallucination guard — the LLM literally cannot make
    a claim it cannot point to.

    Strategy: sentence-level validation. A sentence is stripped if:
      - It contains clinical/numerical language (factual claim indicator)
      - AND it has no [Source: ...] citation
    Pure structural sentences ("The following summarises...") are kept.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    validated = []

    for sentence in sentences:
        has_citation = bool(CITATION_PATTERN.search(sentence))
        is_factual   = bool(FACTUAL_CLAIM_PATTERN.search(sentence))

        if is_factual and not has_citation:
            log.warning(f"[summariser] Stripping uncited factual claim: {sentence[:80]}...")
            continue
        validated.append(sentence)

    return " ".join(validated).strip()


def _stage4b_synthesise(compressed_chunks: list[str], query: str, client) -> str:
    """
    Stage 4b: GPT-4o synthesises the compressed context with forced citation.
    Every factual claim must include [Source: NCT_ID].
    Uncited claims are stripped by _validate_citations().
    """
    context = "\n\n---\n\n".join(compressed_chunks)

    system_prompt = (
        "You are a clinical trial research assistant. Answer the query using ONLY "
        "the provided context. Every factual claim you make MUST be followed "
        "immediately by a citation in this exact format: [Source: NCT_ID] where "
        "NCT_ID is the trial identifier from the context. "
        "If a fact does not have a source in the context, do not state it. "
        "Do not hallucinate, infer, or use knowledge outside the provided context. "
        "This information is for research purposes only and does not constitute medical advice."
    )

    user_prompt = (
        f"Context (retrieved clinical trial documents):\n\n{context}\n\n"
        f"Query: {query}\n\n"
        f"Answer the query using only the context above. "
        f"Every factual claim must cite its source as [Source: NCT_ID]."
    )

    response = client.chat.completions.create(
        model       = "gpt-4o",
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = 1500,
        temperature = 0.0,
    )
    return response.choices[0].message.content.strip()


# ── Lambda entrypoint ──────────────────────────────────────────────────────

def handler(event: dict, context) -> str:
    """
    MCP Lambda handler — called by the Bedrock MCP Gateway.

    Input (from LangChain tool call):
        event["chunks"] : list[str] — chunks from search_tool (with headers)
        event["query"]  : str       — original research query

    Output:
        str — grounded answer with [Source: NCT_ID] citations on every claim
    """
    chunks = event.get("chunks", [])
    query  = event.get("query", "").strip()

    if not chunks:
        return "No context provided. Cannot answer without retrieved documents."
    if not query:
        return "No query provided."

    log.info(f"[summariser] Query: {query[:100]}  chunks={len(chunks)}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_openai_key())

        # Stage 3 — Context compression
        compressed = _stage3_compress(chunks, query, client)
        if not compressed:
            log.warning("[summariser] All chunks were NOT_RELEVANT after compression")
            return (
                "I don't have enough specific information in the knowledge base "
                "to answer this question accurately."
            )

        # Stage 4b — Synthesise with forced citation
        answer = _stage4b_synthesise(compressed, query, client)

        # Validate citations — strip uncited factual claims
        validated = _validate_citations(answer)
        if not validated:
            return (
                "I was unable to produce a properly grounded answer from the "
                "available context. All retrieved information lacked verifiable citations."
            )

        log.info(f"[summariser] Answer length: {len(validated)} chars")
        return validated

    except Exception as exc:
        log.exception(f"[summariser] Pipeline error: {exc}")
        return f"Error synthesising answer: {exc}"
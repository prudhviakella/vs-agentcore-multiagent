"""
mcp_tools/chart_lambda/handler.py
===================================
MCP Lambda tool: generates Chart.js JSON config from numerical clinical trial data.

Registered in MCP Gateway as tool name: chart___chart_tool
Input:  {"data": {...}, "chart_type": "auto", "title": "..."}
Output: {"chart": {...Chart.js config...}, "chart_type": "bar"}

CALL PATH:
  Chart Agent LLM decides to call chart___chart_tool
    → LangChain StructuredTool.invoke()
      → httpx POST (SigV4 signed) → Bedrock MCP Gateway
        → This Lambda
          → Returns Chart.js JSON config
            → Chart Agent app.py detects {"type": "chart"} in tool_end
              → Yields chart event to Supervisor
                → Supervisor re-streams to Platform API
                  → UI renders Chart.js canvas inline in chat bubble

CHART TYPES SUPPORTED:
  bar       → comparisons between trials/drugs (most common)
  line      → trends over time (enrollment, efficacy by date)
  pie       → proportional distributions (Phase 1/2/3/4 split)
  doughnut  → same as pie, better readability for small slices
  auto      → detect from data shape (default)

AUTO-DETECTION LOGIC:
  - Single dataset, categorical labels → bar
  - Single dataset, date/year labels   → line
  - Single dataset, parts of a whole  → doughnut
  - Multiple datasets                  → bar (grouped)

CHART.JS CONFIG STRUCTURE RETURNED:
  {
    "type": "bar",
    "data": {
      "labels": ["BNT162b2 (Pfizer)", "mRNA-1273 (Moderna)", "Ad26.COV2.S (J&J)"],
      "datasets": [{
        "label": "Efficacy %",
        "data": [95.0, 94.1, 66.9],
        "backgroundColor": ["#00c2ff", "#00e5a0", "#ffb340"],
        "borderColor": ["#00c2ff", "#00e5a0", "#ffb340"],
        "borderWidth": 1
      }]
    },
    "options": {
      "responsive": true,
      "plugins": {
        "legend": {"position": "top"},
        "title": {"display": true, "text": "COVID-19 Vaccine Trial Efficacy"}
      },
      "scales": {
        "y": {"beginAtZero": false}
      }
    }
  }

UI RENDERING:
  The UI receives {"type": "chart", "config": {...}} in the SSE stream.
  It creates a <canvas> element and calls new Chart(canvas, config).
  Chart.js renders an interactive chart with hover tooltips inline
  in the chat bubble — no S3, no image generation, no page reload.

SECRET: /vs-agentcore-multiagent/prod/openai → {"api_key": "..."}
  Used for GPT-4o-mini to parse and structure the input data if needed.

COLORS (matches dark clinical UI theme):
  Primary:   #00c2ff  (accent blue)
  Secondary: #00e5a0  (green)
  Tertiary:  #ffb340  (amber)
  Quaternary:#ff4d6a  (red)
  Quinary:   #c084fc  (purple)
"""

import json
import logging
import os
from functools import lru_cache

import boto3

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")

# Clinical UI color palette — matches dark theme
CHART_COLORS = [
    "#00c2ff",  # accent blue
    "#00e5a0",  # green
    "#ffb340",  # amber
    "#ff4d6a",  # red
    "#c084fc",  # purple
    "#64b5f6",  # light blue
    "#a5d6a7",  # light green
    "#ffcc80",  # light amber
]

_oai_client = None


@lru_cache(maxsize=1)
def _get_openai_key() -> str:
    sm = boto3.client("secretsmanager", region_name=REGION)
    return json.loads(
        sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"]
    )["api_key"]


def _get_openai():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        _oai_client = OpenAI(api_key=_get_openai_key())
    return _oai_client


# ── Chart type detection ──────────────────────────────────────────────────────

def _detect_chart_type(data: dict, hint: str = "auto") -> str:
    """
    Detect the best chart type from data shape.

    Args:
        data: The structured chart data dict.
        hint: "auto" | "bar" | "line" | "pie" | "doughnut"

    Returns:
        Chart.js chart type string.
    """
    if hint != "auto":
        return hint

    datasets = data.get("datasets", [])
    labels   = data.get("labels", [])

    # Multiple datasets → grouped bar
    if len(datasets) > 1:
        return "bar"

    # Check if labels look like years/dates → line chart
    if labels and all(
        str(l).isdigit() and 1990 <= int(str(l)) <= 2030
        for l in labels if str(l).isdigit()
    ):
        return "line"

    # Small number of categories (≤6) with single dataset → doughnut
    if len(labels) <= 6 and len(datasets) == 1:
        return "doughnut"

    # Default: bar
    return "bar"


# ── Chart.js config builder ───────────────────────────────────────────────────

def _build_chartjs_config(
    chart_type: str,
    labels:     list,
    datasets:   list[dict],
    title:      str = "",
) -> dict:
    """
    Build a complete Chart.js config dict.

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut"
        labels:     X-axis labels or pie segment names.
        datasets:   List of dataset dicts {label, data, ...}.
        title:      Chart title for the legend.

    Returns:
        Complete Chart.js config ready for new Chart(canvas, config).
    """
    # Assign colors to each dataset or each data point (for pie/doughnut)
    for i, ds in enumerate(datasets):
        if chart_type in ("pie", "doughnut"):
            # Each segment gets its own color
            ds.setdefault("backgroundColor", [
                CHART_COLORS[j % len(CHART_COLORS)]
                for j in range(len(ds.get("data", [])))
            ])
        else:
            color = CHART_COLORS[i % len(CHART_COLORS)]
            ds.setdefault("backgroundColor", color)
            ds.setdefault("borderColor",     color)
            ds.setdefault("borderWidth",     1)
            if chart_type == "line":
                ds.setdefault("fill",   False)
                ds.setdefault("tension", 0.3)

    config = {
        "type": chart_type,
        "data": {
            "labels":   labels,
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"position": "top"},
                "title": {
                    "display": bool(title),
                    "text":    title,
                    "font":    {"size": 14},
                },
                "tooltip": {
                    "mode": "index",
                    "intersect": False,
                },
            },
        },
    }

    # Add axis scales for bar and line (not pie/doughnut)
    if chart_type in ("bar", "line"):
        config["options"]["scales"] = {
            "x": {"grid": {"display": False}},
            "y": {"beginAtZero": chart_type == "bar"},
        }

    return config


# ── LLM data extraction ───────────────────────────────────────────────────────

def _extract_chart_data_via_llm(text: str, chart_type: str) -> dict:
    """
    Use GPT-4o-mini to extract structured chart data from free text.

    Called when the Chart Agent passes raw text instead of structured data.
    Extracts labels, values, title, and dataset label from the text.

    Args:
        text:       Free text containing numerical data.
        chart_type: Target chart type (helps guide extraction).

    Returns:
        Structured chart data dict: {title, labels, datasets}
    """
    oai    = _get_openai()
    prompt = f"""Extract chart data from the following text for a {chart_type} chart.

Text:
{text[:3000]}

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "title": "chart title here",
  "labels": ["label1", "label2", "label3"],
  "datasets": [
    {{
      "label": "dataset label",
      "data": [value1, value2, value3]
    }}
  ]
}}

Rules:
- Extract all numerical values mentioned
- Labels should be short (< 30 chars)
- Values must be numbers (float or int)
- If comparing multiple metrics, use multiple datasets
- Title should describe what is being compared
"""

    try:
        resp = oai.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
            max_tokens  = 500,
        )
        raw  = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        log.error(f"[Chart] LLM extraction failed: {exc}")
        return {
            "title":    "Clinical Trial Data",
            "labels":   ["No data extracted"],
            "datasets": [{"label": "Values", "data": [0]}],
        }


# ── Lambda handler ────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    MCP Lambda handler — generates Chart.js config from chart data.

    event (from Chart Agent via MCP Gateway):
        data       — dict with {title, labels, datasets} OR free text string
        chart_type — "auto" | "bar" | "line" | "pie" | "doughnut" (default: "auto")
        title      — optional chart title override

    Returns:
        {
            "chart":      {...Chart.js config...},
            "chart_type": "bar"
        }
    """
    log.info(f"[Chart Lambda] event keys={list(event.keys())}")

    data       = event.get("data", {})
    chart_type = event.get("chart_type", "auto")
    title      = event.get("title", "")

    try:
        # If data is a string → use LLM to extract structured data
        if isinstance(data, str):
            log.info("[Chart Lambda] Data is text — extracting via LLM")
            extracted  = _extract_chart_data_via_llm(data, chart_type)
            labels     = extracted.get("labels", [])
            datasets   = extracted.get("datasets", [])
            title      = title or extracted.get("title", "Clinical Trial Data")

        # If data is already structured
        elif isinstance(data, dict):
            labels   = data.get("labels",   [])
            datasets = data.get("datasets", [])
            title    = title or data.get("title", "")

        else:
            return {"chart": {}, "error": f"Unsupported data type: {type(data).__name__}"}

        if not labels or not datasets:
            return {"chart": {}, "error": "No labels or datasets found in data"}

        # Detect chart type
        resolved_type = _detect_chart_type(
            {"labels": labels, "datasets": datasets},
            hint=chart_type
        )

        # Build Chart.js config
        config = _build_chartjs_config(
            chart_type = resolved_type,
            labels     = labels,
            datasets   = datasets,
            title      = title,
        )

        log.info(
            f"[Chart Lambda] Generated {resolved_type} chart"
            f"  labels={len(labels)}"
            f"  datasets={len(datasets)}"
            f"  title='{title}'"
        )

        return {
            "chart":      config,
            "chart_type": resolved_type,
        }

    except Exception as exc:
        log.exception(f"[Chart Lambda] Error: {exc}")
        return {"chart": {}, "error": str(exc)}
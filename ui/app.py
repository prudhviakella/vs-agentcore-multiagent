"""
app.py — VS AgentCore UI
=========================
FastAPI app serving a beautiful single-page clinical research UI.
Proxies SSE streams to the Platform API to avoid CORS.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8501
"""

import os
import base64
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

log = logging.getLogger(__name__)

API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")
API_KEY = (os.environ.get("AGENT_API_KEY") or
          os.environ.get("PLATFORM_API_KEY") or
          "local-dev-key")
DOMAIN  = os.environ.get("AGENT_DOMAIN",  "pharma")
AGENT   = "clinical-trial"

HEADERS = {
    "X-API-Key":    API_KEY,
    "Content-Type": "application/json",
    "Accept":       "text/event-stream",
}

app = FastAPI()


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    return await _proxy_sse(f"{API_URL}/api/v1/{AGENT}/chat", body)


@app.post("/resume")
async def resume(request: Request):
    body = await request.json()
    return await _proxy_sse(f"{API_URL}/api/v1/{AGENT}/resume", body)


async def _proxy_sse(url: str, payload: dict):
    async def generate():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, headers=HEADERS, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# Minimal favicon to silence 404 in browser console
_FAVICON = base64.b64decode(
    "AAABAAEAEBAQAAEABAAoAQAAFgAAACgAAAAQAAAAIAAAAAEABAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAA////AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)

@app.get("/favicon.ico")
async def favicon():
    return Response(content=_FAVICON, media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Trial Research Agent</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --font:'AppleGothic','Apple Gothic','Gill Sans MT','Gill Sans','Century Gothic','Trebuchet MS',ui-rounded,sans-serif;
  --bg:#07111e;--surface:#0d1928;--surface-2:#112030;
  --border:#1a2e44;--border-2:#1f3550;
  --accent:#00c2ff;--accent-dim:rgba(0,194,255,0.10);--accent-glow:rgba(0,194,255,0.05);
  --green:#00e5a0;--red:#ff4d6a;--amber:#ffb340;
  --text:#cddff0;--text-2:#6b8fae;--text-3:#2e4a63;
  --r:8px;--r-lg:14px;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);-webkit-font-smoothing:antialiased}
.shell{display:grid;grid-template-rows:58px 1fr auto;height:100vh;max-width:860px;margin:0 auto}

/* header */
header{display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--border);background:var(--surface)}
.brand{display:flex;align-items:center;gap:11px}
.brand-icon{width:30px;height:30px;background:linear-gradient(135deg,#00c2ff,#005fff);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:15px}
.brand-name{font-size:15px;font-weight:500;letter-spacing:.01em}
.header-right{display:flex;gap:10px;align-items:center}
.pill{font-size:11px;font-family:var(--font);color:var(--text-2);background:var(--surface-2);border:1px solid var(--border);padding:3px 10px;border-radius:20px;letter-spacing:.02em}
.pill.live{color:var(--green);border-color:rgba(0,229,160,.25);background:rgba(0,229,160,.06)}

/* messages */
.messages{overflow-y:auto;padding:28px 28px 8px;display:flex;flex-direction:column;gap:20px;scroll-behavior:smooth}
.messages::-webkit-scrollbar{width:3px}
.messages::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:2px}

/* welcome */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;padding:52px 20px;text-align:center}
.welcome-eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:500}
.welcome-title{font-size:36px;font-weight:500;line-height:1.15;letter-spacing:-.01em}
.welcome-title span{color:var(--accent);font-style:italic}
.welcome-sub{font-size:14px;color:var(--text-2);max-width:420px;line-height:1.7;font-weight:300}
.starters-label{font-size:11px;color:var(--text-3);letter-spacing:.06em;text-transform:uppercase}
.starters{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:620px}
.starter{background:var(--surface);border:1px solid var(--border);color:var(--text-2);font-family:var(--font);font-size:13px;font-weight:300;padding:9px 16px;border-radius:22px;cursor:pointer;transition:all .16s;line-height:1.4}
.starter:hover{border-color:var(--accent);color:var(--text);background:var(--accent-glow);transform:translateY(-1px)}

/* messages */
.msg{display:flex;gap:12px;animation:fadeUp .18s ease}
.msg.user{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500}
.av.agent{background:rgba(0,194,255,.12);border:1px solid rgba(0,194,255,.25);color:var(--accent)}
.av.user{background:var(--surface-2);border:1px solid var(--border);color:var(--text-2)}
.bubble{max-width:78%;padding:13px 18px;border-radius:var(--r-lg);font-size:14px;line-height:1.75;font-weight:300}
.msg.user .bubble{background:var(--surface-2);border:1px solid var(--border);color:var(--text);border-radius:var(--r-lg) 4px var(--r-lg) var(--r-lg)}
.msg.agent .bubble{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px var(--r-lg) var(--r-lg) var(--r-lg)}

/* markdown */
.bubble h1,.bubble h2,.bubble h3{font-weight:500;margin:16px 0 8px;letter-spacing:-.01em}
.bubble h1{font-size:18px}.bubble h2{font-size:16px}.bubble h3{font-size:14px}
.bubble p{margin:8px 0}.bubble ul,.bubble ol{padding-left:20px;margin:8px 0}.bubble li{margin:5px 0}
.bubble strong{font-weight:500}.bubble em{color:var(--text-2)}
.bubble code{font-family:'Menlo','Courier New',monospace;font-size:12px;background:var(--surface-2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;color:var(--accent)}
.bubble pre{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:12px;overflow-x:auto;margin:10px 0}
.bubble pre code{background:none;border:none;padding:0;font-size:12px}
.bubble blockquote{border-left:3px solid var(--accent);padding-left:14px;margin:8px 0;color:var(--text-2)}
.bubble a{color:var(--accent);text-decoration:none}
.bubble hr{border:none;border-top:1px solid var(--border);margin:14px 0}
.bubble table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
.bubble table th{background:var(--surface);color:var(--accent);font-weight:600;padding:8px 12px;text-align:left;border-bottom:2px solid var(--border)}
.bubble table td{padding:7px 12px;border-bottom:1px solid var(--border);color:var(--text)}
.bubble table tr:last-child td{border-bottom:none}
.bubble table tr:hover td{background:var(--surface)}
.bubble th,.bubble td{border:1px solid var(--border);padding:7px 12px;text-align:left}
.bubble th{background:var(--surface-2);font-weight:500}
.meta-footer{font-size:11px;color:var(--text-3);margin-top:12px;padding-top:10px;border-top:1px solid var(--border);display:flex;gap:14px;font-weight:300;letter-spacing:.01em}

/* tool step */
.tool-step{display:flex;align-items:center;gap:10px;padding:9px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);font-size:12.5px;font-weight:300;color:var(--text-2);max-width:320px;margin-left:42px;animation:fadeUp .18s ease;letter-spacing:.01em}
.tool-step.done{color:var(--text-2);border-color:var(--border)}
.tool-step.thinking{border-style:dashed;border-color:var(--border-2)}
.spin{width:13px;height:13px;border:1.5px solid var(--border-2);border-top-color:var(--accent);border-radius:50%;animation:spin .65s linear infinite;flex-shrink:0}
.spin.pulse{border-color:transparent;border-top-color:var(--text-3);animation:spin 1.2s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.cur{display:inline-block;width:2px;height:13px;background:var(--accent);margin-left:1px;vertical-align:middle;animation:blink .9s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* hitl */
.hitl{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:var(--r-lg);padding:20px;max-width:500px;margin-left:42px;animation:fadeUp .18s ease}
.hitl-tag{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;display:flex;align-items:center;gap:6px;font-weight:500}
.hitl-tag::before{content:'';width:5px;height:5px;background:var(--accent);border-radius:50%}
.hitl-q{font-size:14px;font-weight:400;color:var(--text);margin-bottom:16px;line-height:1.55}
.hitl-opts{display:flex;flex-direction:column;gap:7px;margin-bottom:12px}
.hitl-opt{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;font-size:13px;font-weight:300;color:var(--text-2);transition:all .14s;text-align:left;width:100%;font-family:var(--font)}
.hitl-opt:hover{border-color:var(--accent);color:var(--text);background:var(--accent-dim);transform:translateX(2px)}
.hitl-opt:hover .num{background:var(--accent);color:var(--bg)}
.hitl-opt.picked{border-color:var(--green);color:var(--green);background:rgba(0,229,160,.07);pointer-events:none}
.num{font-size:11px;width:20px;height:20px;background:var(--border);border-radius:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .14s;font-family:'Menlo',monospace}
.hitl-hint{font-size:12px;color:var(--text-3);font-style:italic;font-weight:300}

/* input */
.bar{padding:16px 28px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:10px;align-items:flex-end}
.inp-wrap{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:var(--r-lg);display:flex;align-items:flex-end;padding:11px 16px;transition:border-color .14s}
.inp-wrap:focus-within{border-color:var(--accent)}
textarea{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--font);font-size:14px;font-weight:300;resize:none;line-height:1.5;max-height:120px;min-height:24px}
textarea::placeholder{color:var(--text-3)}
.send{width:36px;height:36px;background:var(--accent);border:none;border-radius:var(--r);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .14s;color:var(--bg)}
.send:hover{background:#33ccff;transform:scale(1.06)}
.send:disabled{background:var(--border);cursor:not-allowed;transform:none}
.send svg{width:15px;height:15px}
.new-btn{height:36px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;color:var(--text-2);font-family:var(--font);font-size:12px;font-weight:300;padding:0 14px;transition:all .14s;white-space:nowrap}
.new-btn:hover{border-color:var(--border-2);color:var(--text)}
.err{background:rgba(255,77,106,.07);border:1px solid rgba(255,77,106,.25);border-radius:var(--r);padding:10px 14px;font-size:13px;font-weight:300;color:var(--red);margin-left:42px;max-width:460px;animation:fadeUp .18s ease}
.thinking-wrap{margin-left:42px;max-width:640px;margin-bottom:6px;animation:fadeUp .18s ease}
.thinking-toggle{display:flex;align-items:center;gap:7px;background:none;border:1px solid var(--border);border-radius:var(--r);padding:6px 12px;cursor:pointer;color:var(--text-2);font-family:var(--font);font-size:12px;font-weight:300;letter-spacing:.02em;transition:all .14s;width:auto}
.thinking-toggle:hover{border-color:var(--border-2);color:var(--text)}
.thinking-toggle .arr{font-size:10px;transition:transform .2s}
.thinking-toggle.open .arr{transform:rotate(90deg)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0;animation:livepulse 1.1s ease-in-out infinite}
@keyframes livepulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.65)}}
.thinking-body{display:none;margin-top:6px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:12px 16px;font-size:12.5px;font-weight:300;color:var(--text-2);line-height:1.7;white-space:pre-wrap;max-height:300px;overflow-y:auto}
.thinking-body.open{display:block}
.chart-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:20px;margin-left:42px;max-width:640px;animation:fadeUp .18s ease}
.chart-wrap canvas{max-height:320px}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

/* ── message rows ── */
.row{display:flex;align-items:flex-start;gap:10px;margin-bottom:14px;animation:fadeUp .18s ease}
.row-user{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500}
.av-user{background:var(--accent);color:var(--bg)}
.av-ai{background:var(--surface);border:1px solid var(--border);color:var(--text-2)}
.bubble{max-width:640px;padding:12px 16px;border-radius:var(--r-lg);font-size:14px;line-height:1.7;font-weight:300}
.user-bubble{background:var(--surface);border:1px solid var(--border);color:var(--text);border-top-right-radius:4px}
.agent-bubble{background:transparent;color:var(--text);border-top-left-radius:4px}
.agent-bubble p{margin:0 0 10px}
.agent-bubble p:last-child{margin-bottom:0}
.agent-bubble ul,.agent-bubble ol{margin:6px 0 10px 18px;padding:0}
.agent-bubble li{margin-bottom:4px}
.agent-bubble strong{font-weight:500;color:var(--text)}
.agent-bubble code{font-family:'Menlo',monospace;font-size:12.5px;background:var(--surface);padding:2px 5px;border-radius:4px;border:1px solid var(--border)}
/* ── tool steps ── */
.done-tick{color:var(--text-2)}
.tool-step.done .num{color:var(--text-2)}
/* ── meta footer ── */
.meta{display:flex;gap:12px;font-size:11px;color:var(--text-3,#4a6a8a);margin-top:10px;padding-top:8px;border-top:1px solid var(--border)}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="brand-icon">⚕</div>
      <span class="brand-name">Clinical Trial Research Agent</span>
    </div>
    <div class="header-right">
      <span class="pill" id="sid">session: —</span>
      <span class="pill live">● pharma</span>
    </div>
  </header>

  <div class="messages" id="msgs">
    <div class="welcome" id="welcome">
      <div class="welcome-eyebrow">Vidya Sankalp · AgentCore Platform</div>
      <div class="welcome-title">Clinical Trial<br><span>Intelligence</span></div>
      <p class="welcome-sub">Search 5,772 trial documents and a live biomedical knowledge graph. Powered by Pinecone, Neo4j, and AWS AgentCore.</p>
      <div class="starters-label">Try one of these</div>
      <div class="starters" id="starters"></div>
    </div>
  </div>

  <div class="bar">
    <div class="inp-wrap">
      <textarea id="inp" placeholder="Ask about a trial, drug, or clinical outcome…" rows="1"></textarea>
    </div>
    <button class="send" id="sbtn" onclick="send()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
    <button class="new-btn" onclick="newChat()">+ New</button>
  </div>
</div>

<script>
const STARTERS = [
  "What are the Phase 3 results for the mRNA-1273 vaccine?",
  "Which cancer trials are in the knowledge base?",
  "What were the efficacy results?",
  "Compare efficacy across COVID-19 vaccine trials and show me a chart",
  "What are the primary endpoints of NCT04470427?",
  "Which trials are in Phase 3 and currently active?",
];

// UUID fallback — crypto.randomUUID() only works on HTTPS (secure context)
// The ALB runs on HTTP so we need a Math.random() fallback
function uuid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

// inp declared first so send() can reference it without hoisting issues
const inp = document.getElementById('inp');

let threadId    = uuid();
let interrupted = false;
let streaming   = false;

document.getElementById('sid').textContent = 'session: ' + threadId.slice(0, 8);

// Starter chips
const startersEl = document.getElementById('starters');
STARTERS.forEach(q => {
  const b = document.createElement('button');
  b.className = 'starter';
  b.textContent = q;
  b.onclick = () => submit(q);
  startersEl.appendChild(b);
});

// Auto-resize textarea
inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
});
inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// ── Send ──────────────────────────────────────────────────────────────────
function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

function send() {
  const t = inp.value.trim();
  if (!t || streaming) return;
  inp.value = ''; inp.style.height = 'auto';
  submit(t);
}

async function submit(text) {
  hideWelcome();
  addUser(text);
  if (interrupted) await doResume(text);
  else await doChat(text);
}

async function doChat(msg) {
  await sse('/chat', { message: msg, thread_id: threadId, domain: 'pharma' });
}
async function doResume(ans) {
  interrupted = false;
  await sse('/resume', { thread_id: threadId, user_answer: ans, domain: 'pharma' });
}

// ── SSE stream ────────────────────────────────────────────────────────────
// ── State machine for token routing ─────────────────────────────────────────
//
// SSE patterns observed across all query types:
//
// PATTERN A — Late-close multi-thinking (most common, non-chart)
//   <thinking>plan + sub-agent calls + responses</thinking>
//   <thinking>replan</thinking>Note: Final answer...
//   done
//
// PATTERN B — Early-close with chart
//   <thinking>just the plan</thinking>
//   multi_tool_use.parallel({...})   ← tool call JSON as tokens
//   tool_start / tool_end            ← agent tools
//   sub-agent response tokens
//   chart event
//   tool_end
//   Here is a chart...               ← final answer tokens
//   done
//
// PATTERN C — Resume / plain (no thinking)
//   tool_start / tool_end
//   Answer tokens directly
//   done
//
// States:
//   THINKING  inThinking=true    accumulate in blockBuf, shown in reasoning card
//   PREBUF    !inThinking&&!started  buffer tokens looking for <thinking>
//     - discarding=true: drop junk, keep last 12 chars for tag detection
//                        AND accumulate in lastAnswerBuf (for done-flush)
//     - discarding=false: accumulate in prebuf (plain answer buffer)
//   ANSWER    started=true       stream tokens to answer bubble
//
async function sse(url, payload) {
  streaming = true;
  setDisabled(true);

  // ── State — each variable has ONE purpose ────────────────────────────────
  let agentEl      = null;   // answer bubble DOM element
  let content      = '';     // accumulated answer text
  let latency      = 0;
  let started      = false;  // answer bubble has been opened

  // Thinking card state
  let cardEl       = null;   // single reasoning card (never recreated once made)
  let cardText     = '';     // full cumulative text in card (grows, never reset)
  let blockBuf     = '';     // current <thinking> block being streamed (reset per block)
  let inThinking   = false;  // currently inside <thinking>...</thinking>

  // Pre-buffer — accumulates tokens until we know where to route them
  let prebuf       = '';

  // Tool / flow control
  let toolActive     = false; // between tool_start and tool_end
  let discarding     = false; // junk tokens — drop them
  let readyForAnswer = false; // chart fired — next non-thinking content is the answer
  let lastAnswerBuf  = '';    // fallback: tokens captured during tool execution

  // ── Persistent status indicator — always visible while processing ─────────
  // One spinner that stays until the answer starts or an error occurs
  const statusEl = addToolStep('Analysing…', true);

  // ── Thinking card helpers ─────────────────────────────────────────────────
  function openThinkingBlock() {
    if (!cardEl) {
      cardEl = addThinkingCard();
      cardText = '';
    } else {
      if (cardText) cardText += '\n\n—\n\n';
      reopenThinkingCard(cardEl);
    }
    blockBuf   = '';
    inThinking = true;
  }

  function appendThinking(chunk) {
    blockBuf += chunk;
    // Strip partial closing tag from live display
    const display = (cardText + blockBuf).replace(/<\/?t(h(i(n(k(i(n(g?)?)?)?)?)?)?)?>?$/i, '');
    updateThinkingCard(cardEl, display);
  }

  function sealThinkingBlock() {
    cardText  += blockBuf;
    blockBuf   = '';
    inThinking = false;
    updateThinkingCard(cardEl, cardText);
    sealThinkingCard(cardEl);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  const isToolCallJson = s => (
    /^[\s\n]*(multi_tool_use|functions\.|\{|\[)/.test(s) ||
    /^[\s\n]*(Let me (use|call|check|retrieve|query|now|first)|I will (use|call|now|first)|I\'ll (use|call|now)|I need to (use|call|retrieve))/i.test(s)
  );

  function removeStatus() {
    if (statusEl && statusEl.parentNode) statusEl.remove();
  }

  function startAnswer(firstChunk) {
    removeStatus();
    agentEl  = addAgentBubble();
    started  = true;
    content += firstChunk;
    streamToken(agentEl, content);
  }

  function handleAfterThinking(afterRaw) {
    const after = afterRaw.trim();

    // Chart already fired — any non-junk content is the final answer
    if (readyForAnswer) {
      readyForAnswer = false;
      if (after && !after.startsWith('<') && !isToolCallJson(afterRaw)) {
        startAnswer(after);
      }
      // else: answer will arrive on next token in PREBUF
      return;
    }

    if (!after || isToolCallJson(afterRaw)) {
      discarding = true; prebuf = after.slice(-12);
      return;
    }
    if (after.startsWith('<thinking>')) {
      openThinkingBlock();
      const rest = after.slice('<thinking>'.length);
      if (rest.includes('</thinking>')) {
        const [blk, ...tail] = rest.split('</thinking>');
        blockBuf = blk;
        sealThinkingBlock();
        lastAnswerBuf = '';
        handleAfterThinking(tail.join('</thinking>'));
      } else {
        appendThinking(rest);
      }
      return;
    }
    if (after.startsWith('<') || /^(Case [A-D]|Where |Target:|Plan:|Q[123]\.|Note:|Step )/.test(after)) {
      discarding = true; prebuf = after.slice(-12);
      return;
    }
    startAnswer(after);
  }

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify(payload),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   rawBuf  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      rawBuf += decoder.decode(value, { stream: true });
      const lines = rawBuf.split('\n');
      rawBuf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        let ev; try { ev = JSON.parse(raw); } catch { continue; }
        const t = ev.type || '';

        // ── Events ────────────────────────────────────────────────────────
        if (t === 'tool_start') {
          // Update spinner label — don't remove it
          if (statusEl && statusEl.parentNode) {
            statusEl.querySelector('span:last-child').textContent = 'Analysing…';
          }
          toolActive = true;
          if (started && agentEl) {
            const row = agentEl.closest('.row');
            if (row) row.remove();
            agentEl = null; started = false; content = '';
          }
          lastAnswerBuf = '';
          continue;
        }

        if (t === 'tool_end') {
          toolActive    = false;
          lastAnswerBuf = '';
          continue;
        }

        if (t === 'interrupt') {
          removeStatus();
          lastAnswerBuf = '';
          prebuf        = '';
          if (agentEl) {
            const row = agentEl.closest('.row');
            if (row) row.remove();
            agentEl = null; started = false; content = '';
          }
          addHITL(ev.question || 'Please clarify:', ev.options || [], ev.allow_freetext !== false);
          // Add "Clarification needed" indicator
          const cl = addToolStep('Clarification needed', false);
          cl.classList.add('done');
          interrupted = true; streaming = false; setDisabled(false); return;
        }

        if (t === 'chart') {
          addChart(ev.config);
          readyForAnswer = true;
          lastAnswerBuf  = '';
          prebuf         = '';
          if (!inThinking) discarding = false;
          // Add "Chart ready" indicator
          const cr = addToolStep('Chart ready', false);
          cr.classList.add('done');
          continue;
        }

        if (t === 'error') {
          removeStatus();
          addErr(ev.message || 'Unknown error');
          break;
        }

        if (t === 'done') {
          latency = ev.latency_ms || 0;
          if (!started && !inThinking) {
            const answer = (prebuf.trim() && !discarding) ? prebuf.trim() : lastAnswerBuf.trim();
            if (answer) startAnswer(answer);
          }
          if (inThinking && cardEl) sealThinkingBlock();
          continue;
        }

        // ── Token ─────────────────────────────────────────────────────────
        const token = ev.content || ev.result || ev.token || '';
        if (!token) continue;

        // Base64 image from chart agent
        if (token.includes('data:image/') && token.includes('base64,')) {
          if (!agentEl) { agentEl = addAgentBubble(); started = true; }
          const wrap = document.createElement('div');
          wrap.style.cssText = 'margin:12px 0;max-width:100%';
          const img  = document.createElement('img');
          const b64  = token.match(/data:image\/[^;]+;base64,[A-Za-z0-9+\/=]+/);
          if (b64) img.src = b64[0];
          img.style.cssText = 'max-width:100%;border-radius:8px;border:1px solid var(--border)';
          img.alt = 'Chart';
          wrap.appendChild(img);
          agentEl.appendChild(wrap);
          scrollEnd();
          continue;
        }

        // ── THINKING state ────────────────────────────────────────────────
        if (inThinking) {
          blockBuf += token;
          if (!blockBuf.includes('</thinking>')) {
            appendThinking('');   // re-render display (blockBuf already updated)
            continue;
          }
          const [blk, ...tail] = blockBuf.split('</thinking>');
          blockBuf = blk;
          sealThinkingBlock();
          lastAnswerBuf = '';
          handleAfterThinking(tail.join('</thinking>'));
          continue;
        }

        // Discard tokens during tool execution (AFTER inThinking check)
        if (toolActive) { lastAnswerBuf += token; continue; }

        // ── ANSWER state ──────────────────────────────────────────────────
        if (started) {
          content += token;
          streamToken(agentEl, content);
          continue;
        }

        // ── PREBUF state ──────────────────────────────────────────────────
        // readyForAnswer: chart fired, waiting for next thinking or answer
        if (readyForAnswer) {
          prebuf += token;
          if (prebuf.includes('<thinking>')) {
            readyForAnswer = false;
            const [, ...rest] = prebuf.split('<thinking>');
            openThinkingBlock();
            const blockStart = rest.join('<thinking>');
            prebuf = '';
            if (blockStart.includes('</thinking>')) {
              const [blk, ...tail] = blockStart.split('</thinking>');
              blockBuf = blk;
              sealThinkingBlock();
              lastAnswerBuf = '';
              handleAfterThinking(tail.join('</thinking>'));
            } else {
              appendThinking(blockStart);
            }
          } else if (!prebuf.trimStart().startsWith('<') || prebuf.length > 30) {
            readyForAnswer = false;
            startAnswer(prebuf);
            prebuf = '';
          }
          continue;
        }

        prebuf += token;

        // Detect <thinking> open tag
        if (prebuf.includes('<thinking>')) {
          discarding = false;
          const [, ...rest] = prebuf.split('<thinking>');
          openThinkingBlock();
          const blockStart = rest.join('<thinking>');
          prebuf = '';
          if (blockStart.includes('</thinking>')) {
            const [blk, ...tail] = blockStart.split('</thinking>');
            blockBuf = blk;
            sealThinkingBlock();
            lastAnswerBuf = '';
            handleAfterThinking(tail.join('</thinking>'));
          } else {
            appendThinking(blockStart);
          }
          continue;
        }

        if (discarding) {
          prebuf = prebuf.slice(-12);
          lastAnswerBuf += token;
          continue;
        }

        // Plain answer accumulating (resume flow, short direct responses)
        if (prebuf.length > 500) {
          startAnswer(prebuf);
          prebuf = '';
        }
      }
    }

    if (started && agentEl) {
      const cleaned = clean(content);
      if (!cleaned) addErr('Response could not be displayed — try + New for a fresh session.');
      else finalize(agentEl, cleaned, latency);
    } else if (!started && !interrupted) {
      addErr('No response received.');
    }

  } catch (e) {
    addErr('Connection error: ' + e.message);
  } finally {
    removeStatus();
    streaming = false;
    setDisabled(false);
    scrollEnd();
  }
}


// ── DOM helpers ───────────────────────────────────────────────────────────
function scrollEnd() {
  const m = document.getElementById('msgs');
  m.scrollTop = m.scrollHeight;
}

function setDisabled(on) {
  document.getElementById('inp').disabled  = on;
  document.getElementById('sbtn').disabled = on;
}

function newChat() {
  if (streaming) return;
  threadId    = uuid();
  interrupted = false;
  document.getElementById('msgs').innerHTML = '';
  document.getElementById('sid').textContent = 'session: ' + threadId.slice(0,8);
  const w = document.createElement('div');
  w.className = 'welcome';
  w.id = 'welcome';
  w.innerHTML = '<div class="welcome-eyebrow">Vidya Sankalp · AgentCore Platform</div>' +
    '<div class="welcome-title">Clinical Trial<br><span>Intelligence</span></div>' +
    '<p class="welcome-sub">Search 5,772 trial documents and a live biomedical knowledge graph. Powered by Pinecone, Neo4j, and AWS AgentCore.</p>' +
    '<div class="starters-label">Try one of these</div>' +
    '<div class="starters" id="starters"></div>';
  document.getElementById('msgs').appendChild(w);
  const startersEl2 = document.getElementById('starters');
  STARTERS.forEach(q => {
    const b = document.createElement('button');
    b.className = 'starter'; b.textContent = q;
    b.onclick = () => submit(q);
    startersEl2.appendChild(b);
  });
}

function addUser(text) {
  const m = document.getElementById('msgs');
  const row = document.createElement('div');
  row.className = 'row row-user';
  row.innerHTML = '<div class="bubble user-bubble">' + esc(text) + '</div><div class="av av-user">U</div>';
  m.appendChild(row); scrollEnd();
}

function addAgentBubble() {
  const m = document.getElementById('msgs');
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = '<div class="av av-ai">AI</div><div class="bubble agent-bubble"><span class="cur"></span></div>';
  m.appendChild(row); scrollEnd();
  return row.querySelector('.agent-bubble');
}

function streamToken(el, rawContent) {
  // Show cursor while streaming; replace content preserving cursor
  el.innerHTML = mdParse(rawContent) + '<span class="cur"></span>';
  scrollEnd();
}

function finalize(el, cleaned, latency_ms) {
  el.innerHTML = mdParse(cleaned);
  const meta = document.createElement('div');
  meta.className = 'meta';
  const s = (latency_ms / 1000).toFixed(1);
  meta.innerHTML = '<span>&#9201; ' + s + 's</span><span>Research purposes only &middot; Not medical advice</span>';
  el.appendChild(meta);
  scrollEnd();
}

function addToolStep(label, active) {
  const m   = document.getElementById('msgs');
  const el  = document.createElement('div');
  el.className = 'tool-step' + (active ? ' thinking' : '');
  el.innerHTML = (active
    ? '<span class="spin"></span>'
    : '<span class="num">✓</span>') + '<span>' + esc(label) + '</span>';
  m.appendChild(el); scrollEnd();
  return el;
}

function markDone(el, label) {
  if (!el) return;
  // Hide generic analysis steps — only keep chart/clarification visible
  const keep = label === 'Chart ready' || label === 'Clarification needed';
  if (!keep) {
    el.remove();
    return;
  }
  el.classList.add('done');
  el.classList.remove('thinking');
  el.innerHTML = '<span class="num done-tick">✓</span><span>' + esc(label) + '</span>';
}

function addErr(msg) {
  const m  = document.getElementById('msgs');
  const el = document.createElement('div');
  el.className = 'err';
  el.textContent = '⚠ ' + msg;
  m.appendChild(el); scrollEnd();
}

function addHITL(question, options, allowFreetext) {
  const m  = document.getElementById('msgs');
  const el = document.createElement('div');
  el.className = 'hitl-card';
  let html = '<div class="hitl-tag">CLARIFICATION NEEDED</div>';
  html += '<p class="hitl-q">' + esc(question) + '</p>';
  html += '<div class="hitl-opts">';
  options.forEach((opt, i) => {
    html += '<button class="hitl-opt" onclick="chooseHITL(this,' + "'" + escA(opt) + "'" + ')">' +
            '<span class="hitl-num">' + (i+1) + '</span><span>' + esc(opt) + '</span></button>';
  });
  html += '</div>';
  if (allowFreetext) {
    html += '<p class="hitl-free">Or type your answer below</p>';
  }
  el.innerHTML = html;
  m.appendChild(el); scrollEnd();
}

function chooseHITL(btn, opt) {
  // Highlight selected option in green, disable all buttons in this card
  const card = btn.closest('.hitl-card');
  if (card) {
    card.querySelectorAll('.hitl-opt').forEach(b => {
      b.style.pointerEvents = 'none';
      b.style.opacity = '0.45';
    });
    btn.classList.add('picked');
    btn.style.opacity = '1';
    // Hide free-text hint
    const free = card.querySelector('.hitl-free');
    if (free) free.style.display = 'none';
  }
  submit(opt);
}

function mdParseTable(block) {
  // Parse a markdown table block into HTML
  const lines = block.trim().split('\n').filter(l => l.trim());
  if (lines.length < 2) return null;
  // Check second line is separator (---|---)
  if (!/^\|?[\s\-|:]+\|?$/.test(lines[1])) return null;
  const parseRow = (line) =>
    line.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  const headers = parseRow(lines[0]);
  const rows    = lines.slice(2).map(parseRow);
  let html = '<table><thead><tr>';
  headers.forEach(h => { html += '<th>' + h + '</th>'; });
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    row.forEach(cell => { html += '<td>' + cell + '</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

function mdParse(t) {
  if (!t) return '';
  t = clean(t);
  // Markdown tables — detect blocks with | characters
  t = t.replace(/((?:\|.+\|\n?){2,})/g, (match) => {
    const parsed = mdParseTable(match);
    return parsed || match;
  });
  // Bold
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Inline code
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Numbered list
  t = t.replace(/^(\d+)\. (.+)$/gm, '<li data-n="$1">$2</li>');
  t = t.replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ol>$&</ol>');
  // Bullet list
  t = t.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  t = t.replace(/(<li>.*<\/li>\n?)+/g, m => m.startsWith('<ol>') ? m : '<ul>' + m + '</ul>');
  // Line breaks → paragraphs
  t = t.split(/\n\n+/).map(p => p.trim()).filter(Boolean).map(p => {
    if (p.startsWith('<ol>') || p.startsWith('<ul>') || p.startsWith('<li') || p.startsWith('<table')) return p;
    return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
  }).join('');
  return t;
}


// ── Thinking card ────────────────────────────────────────────────────────
function addThinkingCard() {
  const m = document.getElementById('msgs');
  const wrap = document.createElement('div');
  wrap.className = 'thinking-wrap';
  wrap.innerHTML =
    '<button class="thinking-toggle streaming" onclick="toggleThinking(this)">' +
    '<span class="arr">▶</span><span>Reasoning</span><span class="live-dot"></span>' +
    '</button>' +
    '<div class="thinking-body"></div>';
  m.appendChild(wrap); scrollEnd();
  return wrap;
}
function updateThinkingCard(wrap, text) {
  if (!wrap) return;
  const body = wrap.querySelector('.thinking-body');
  if (body) body.textContent = text;
  scrollEnd();
}
function sealThinkingCard(wrap) {
  if (!wrap) return;
  const btn = wrap.querySelector('.thinking-toggle');
  if (btn) {
    btn.classList.remove('streaming');
    const dot = btn.querySelector('.live-dot');
    if (dot) dot.style.display = 'none';  // hide dot but keep element for re-opening
    const label = btn.querySelector('span:nth-child(2)');
    if (label) label.textContent = 'Reasoning (' + wrap.querySelector('.thinking-body').textContent.length + ' chars)';
  }
}

function reopenThinkingCard(wrap) {
  if (!wrap) return;
  const btn = wrap.querySelector('.thinking-toggle');
  if (btn) {
    btn.classList.add('streaming');
    const dot = btn.querySelector('.live-dot');
    if (dot) dot.style.display = '';
    const label = btn.querySelector('span:nth-child(2)');
    if (label) label.textContent = 'Reasoning';
  }
}
function toggleThinking(btn) {
  btn.classList.toggle('open');
  const body = btn.nextElementSibling;
  body.classList.toggle('open');
}

// ── Chart rendering ──────────────────────────────────────────────────────
let _chartCount = 0;
function addChart(config) {
  if (!config) return;
  const m = document.getElementById('msgs');
  const wrap = document.createElement('div');
  wrap.className = 'chart-wrap';
  const id = 'chart-' + (++_chartCount);
  wrap.innerHTML = '<canvas id="' + id + '"></canvas>';
  m.appendChild(wrap);
  scrollEnd();

  // Chart.js v4 compat: "horizontalBar" was removed — use bar + indexAxis:'y'
  let chartType = config.type || 'bar';
  if (chartType === 'horizontalBar') {
    chartType = 'bar';
    config.options = config.options || {};
    config.options.indexAxis = 'y';
  }

  // Apply theme colors to match the UI
  config.options = config.options || {};
  config.options.plugins = config.options.plugins || {};
  config.options.plugins.legend = config.options.plugins.legend || {};
  config.options.plugins.legend.labels = { color: '#cddff0', font: { family: 'AppleGothic, sans-serif' } };
  if (config.options.plugins.title) config.options.plugins.title.color = '#cddff0';
  // Pie/doughnut charts don't use scales — applying scales breaks them
  if (!['pie', 'doughnut'].includes(chartType)) {
    config.options.scales = config.options.scales || {};
    ['x','y'].forEach(ax => {
      config.options.scales[ax] = config.options.scales[ax] || {};
      config.options.scales[ax].ticks = {
        color: '#6b8fae',
        maxRotation: ax === 'x' ? 45 : 0,
        minRotation: ax === 'x' ? 45 : 0,
        autoSkip: true,
        maxTicksLimit: ax === 'x' ? 10 : 8,
      };
      config.options.scales[ax].grid = { color: 'rgba(26,46,68,0.8)' };
    });
  }

  new Chart(document.getElementById(id), { type: chartType, data: config.data, options: config.options });
}

// ── Utils ─────────────────────────────────────────────────────────────────
function toolLabel(n) {
  if (!n) return 'Analysing…';
  if (n.includes('chart'))      return 'Chart ready';
  if (n.includes('ask_user'))   return 'Clarification needed';
  return 'Analysing…';
}
function clean(t) {
  if (!t) return '';
  // Strip parallel tool call artifacts GPT-4o leaks as tokens
  t = t.replace(/multi_tool_use\.parallel\(\{[\s\S]*?\}\)/g, '').trim();
  t = t.replace(/functions\.[a-z_]+\(\{[\s\S]*?\}\)/g, '').trim();
  // Strip attachment:// image placeholders from chart agent
  t = t.replace(/!\[[^\]]*\]\(attachment:\/\/[^)]*\)/g, '').trim();
  t = t.replace(/\(attachment:\/\/[^)]*\)/g, '').trim();
  // Strip any <thinking>...</thinking> blocks that leaked into final answer
  t = t.replace(/<thinking>[\s\S]*?<\/thinking>/g, '');
  // Strip any orphaned/partial thinking tags that survived the block stripper
  t = t.replace(/<\/thinking>/g, '');
  t = t.replace(/<thinking>/g, '');
  // Strip Chart.js placeholder text in all forms the LLM emits:
  // ![Chart.js visualisation rendered by chart_tool]
  // [Interactive Chart.js visualisation]
  // ![Efficacy Comparison...](rendered-chart)
  t = t.replace(/!?\[(?:Interactive )?Chart\.js[^\]]*\](?:\([^)]*\))?/gi, '');
  t = t.replace(/!?\[[^\]]*chart[^\]]*\](?:\([^)]*\))?/gi, '');
  t = t.replace(/\n?EPISODIC:\s*(YES|NO)[\d.\s]*/gi, '');
  t = t.replace(/\n?This information is for research purposes only and does not constitute medical advice\.?\s*/gi, '');
  t = t.replace(/\n?\[Reason logged for review:.*?\]\s*/gis, '');
  t = t.replace(/^[-•]\s+(User (inquired|asked|mentioned|expressed)|AI (provided|gave|discussed)|The (user|AI|agent)|Overall,|Key points)[^\n]*/gim, '');
  const summaryStart = /^(User (inquired|asked|mentioned)|A tool was called|The (user|AI|agent) (inquired|asked|provided|discussed)|Here is a summary)/i;
  if (summaryStart.test(t.trim())) {
    const parts = t.split(/\n\n+/);
    const realContent = parts.find(p =>
      p.trim().length > 0 &&
      !summaryStart.test(p.trim()) &&
      !/^[-•]\s+(User|AI|The user|The agent)/i.test(p.trim())
    );
    if (realContent) t = parts.slice(parts.indexOf(realContent)).join('\n\n');
    else return '';
  }
  // Strip base64 image data — chart agent returns PNG as data URI token
  t = t.replace(/data:image\/[^;]+;base64,[A-Za-z0-9+\/=]{20,}/g, '');
  t = t.replace(/!?\[.*?\]\(data:image[^)]*\)/g, '');

  // Remove duplicate paragraphs (sub-agent response echoed in supervisor summary)
  const paras = t.split(/\n\n+/);
  const seen = new Set();
  const deduped = paras.filter(p => {
    const key = p.trim().slice(0, 80).toLowerCase();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  t = deduped.join('\n\n');

  return t.trim();
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escA(s) { return String(s).replace(/'/g, "\\'"); }
</script>
</body>
</html>"""
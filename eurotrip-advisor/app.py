import os, re, pathlib, requests
from typing import Optional, List, Tuple
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Setup --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Travel Advisor", page_icon="🧭", layout="wide")

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------- CSS (high-visibility UI) --------------------
st.markdown("""
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --text:#111827; --muted:#4b5563;
  --border:#e5e7eb; --primary:#111827; --primaryHover:#0d9488;
  --teal:#0f766e; --tealHover:#115e59; --indigo:#4338ca; --indigoHover:#3730a3;
}
html, body, .stApp{background:var(--bg)!important; color:var(--text)!important;}
.block-container{max-width:1100px; margin:auto;}

/* Sidebar: clearer text, white background, strong buttons */
[data-testid="stSidebar"]{
  background:#ffffff!important;
  border-right:1px solid var(--border)!important;
  color:var(--text)!important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] div{ color:var(--text)!important; }

[data-testid="stSidebar"] .stButton>button{
  background:var(--teal)!important;
  border:1px solid var(--teal)!important;
  color:#fff!important;
  border-radius:10px!important;
  font-weight:700!important;
  height:42px;
}
[data-testid="stSidebar"] .stButton>button:hover{
  background:var(--tealHover)!important; border-color:var(--tealHover)!important;
}

/* Inputs & main send button */
.stTextArea textarea, .stTextInput input{
  background:var(--panel)!important; color:var(--text)!important;
  border:1px solid var(--border)!important; border-radius:10px!important;
  font-size:1rem!important; caret-color:#111827!important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder{color:#9ca3af!important;}

.stButton>button{
  background:var(--primary)!important; color:#fff!important;
  border:1px solid var(--primary)!important; border-radius:10px!important;
  font-weight:800!important; height:50px;
}
.stButton>button:hover{background:var(--primaryHover)!important; border-color:var(--primaryHover)!important;}

/* Cards & badges */
.card{background:#fff; border:1px solid var(--border); border-radius:16px; padding:18px;}
.badge{padding:4px 10px; border-radius:999px; border:1px solid var(--border); font-size:.8rem;}
.badge-hit{background:#dcfce7; color:#065f46; border-color:#bbf7d0;}
.badge-fb{background:#ffe4e6; color:#9f1239; border-color:#fecdd3;}
.small{color:var(--muted); font-size:.92rem;}

/* Download button: strong indigo */
div.stDownloadButton > button{
  background:var(--indigo)!important;
  border:1px solid var(--indigo)!important;
  color:#fff!important; font-weight:800!important;
  border-radius:10px!important; height:44px;
}
div.stDownloadButton > button:hover{
  background:var(--indigoHover)!important; border-color:var(--indigoHover)!important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Wikipedia helpers --------------------
HEADERS = {"User-Agent": "Travel-Advisor/1.0 (+streamlit)"}

def fetch_wiki(title: str) -> Optional[str]:
    slug = title.strip().replace(" ", "_")
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/html/{slug}",
            headers=HEADERS, timeout=15
        )
        if r.status_code != 200:
            return None
        from bs4 import BeautifulSoup  # local import fine
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator="\n")
        return f"# {title}\n\n*Source: Wikipedia (CC BY-SA 4.0)*\n\n{text}"
    except Exception:
        return None

def ensure_cache(title: str) -> Optional[pathlib.Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip())
    p = CACHE_DIR / f"{safe}.md"
    if p.exists() and p.stat().st_size > 100:
        return p
    c = fetch_wiki(title)
    if c:
        p.write_text(c, encoding="utf-8")
        return p
    return None

def _read_text(p: pathlib.Path, max_chars=4000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""

# -------------------- Retrieval (strict KB) --------------------
def retrieve_kb_snippets(query: str, top_k=2) -> Tuple[List[str], bool, List[str]]:
    files = list(CACHE_DIR.glob("*.md"))
    if not files:
        return [], False, []

    query_clean = re.sub(r"[^a-zA-Z0-9]+", " ", query.lower()).strip()
    q_words = [w for w in query_clean.split() if len(w) > 3]

    matched = []
    for p in files:
        name = p.stem.lower().replace("_", " ")
        if any(name == w or w in name for w in q_words):
            matched.append(p)

    if not matched:
        return [], False, []

    matched = matched[:top_k]
    snippets, used_names = [], []
    for p in matched:
        used_names.append(p.stem)
        snippets.append(f"[{p.stem}]\n{_read_text(p, 3000)[:1200]}")
    return snippets, True, used_names

# -------------------- OpenAI --------------------
def ask_openai_with_kb(question: str) -> Tuple[str, bool, List[str]]:
    if not client.api_key:
        return "Missing OpenAI API key.", False, []

    snippets, used_kb, used_files = retrieve_kb_snippets(question)
    kb_text = "\n\n".join(snippets) if snippets else ""

    system_prompt = (
        "You are a friendly Europe travel assistant. "
        "If context is provided under 'Context from Wikipedia', use it for facts and naming; "
        "otherwise answer from general travel knowledge. "
        "Do not mention missing sources or heads-up messages yourself; just answer helpfully. "
        "Keep answers concise (2–6 short paragraphs) and practical."
    )

    if used_kb:
        user_prompt = f"Context from Wikipedia:\n{kb_text}\n\nQuestion: {question}"
    else:
        user_prompt = f"Question: {question}"

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    ans = res.choices[0].message.content.strip()

    if not used_kb:
        ans = (
            "Heads up: I couldn’t find relevant info in the indexed Wikipedia pages; "
            "here’s a general best-effort answer.\n\n" + ans
        )

    return ans, used_kb, used_files

# -------------------- State --------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if st.session_state.clear_input:
    st.session_state["q_box"] = ""
    st.session_state.clear_input = False

# -------------------- Header --------------------
st.markdown("<h1>🧭 Travel Advisor — Travel Q&A (from Wikipedia)</h1>", unsafe_allow_html=True)
st.caption("Citations when available • Friendly itineraries • Smart fallback")

# -------------------- Sidebar (visible KB + History) --------------------
with st.sidebar:
    st.subheader("Knowledge Base")
    st.caption("France, Italy, Germany, Spain, Greece, Portugal, Switzerland, Austria…")

    new_page = st.text_input("Add a Wikipedia page", placeholder="e.g., Florence, Santorini")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch & Add", use_container_width=True):
            if new_page.strip():
                p = ensure_cache(new_page.strip())
                st.success(f"Added: {p.name}" if p else "Could not fetch page.")
            else:
                st.warning("Enter a title first.")
    with colB:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history.clear()
            st.toast("History cleared.")

    st.markdown("---")
    st.subheader("History")
    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"- **{item['question']}**", unsafe_allow_html=True)

# -------------------- Input --------------------
st.subheader("Ask a question")
q = st.text_area("Your question", key="q_box",
                 placeholder="e.g., Plan a 5-day Portugal road trip.",
                 height=90)
send = st.button("Send", use_container_width=True, disabled=st.session_state.running)

if send:
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.running = True
        with st.spinner("Thinking..."):
            try:
                ans, kb, used_files = ask_openai_with_kb(q.strip())
            except Exception as e:
                ans, kb, used_files = (f"Error: {e}", False, [])
            st.session_state.history.append({
                "question": q.strip(), "answer": ans, "kb": kb, "sources": used_files
            })
        st.session_state.running = False
        st.session_state.clear_input = True
        st.rerun()

# -------------------- Answer --------------------
if st.session_state.history:
    last = st.session_state.history[-1]
    badge = "KB Hit" if last["kb"] else "Fallback"
    bclass = "badge-hit" if last["kb"] else "badge-fb"
    st.markdown(
        f"<div class='card'><b>Question</b> "
        f"<span class='badge {bclass}'>{badge}</span><br>{last['question']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(last["answer"])
    if last["kb"] and last.get("sources"):
        st.markdown(f"<div class='small'>Source Mode: KB Hit (files: {', '.join(last['sources'])})</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='small'>Source Mode: Fallback</div>", unsafe_allow_html=True)

    st.download_button(
        "Download Answer (Markdown)",
        data=f"# {last['question']}\n\n{last['answer']}",
        file_name="answer.md",
        mime="text/markdown"
    )
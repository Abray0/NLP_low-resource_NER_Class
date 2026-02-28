"""
Arabic NLP Pipeline — Streamlit Frontend
Connects to the FastAPI backend (main.py) for:
  - Named Entity Recognition  → POST /api/ner
  - Text Classification       → POST /api/classification
  - Health check              → GET  /health
"""

import time
from collections import Counter

import requests
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic NLP Pipeline",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&family=Amiri:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #080c10;
    color: #ddd4c0;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #6a7a8a !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0f151c;
    border: 1px solid #1e2530;
    border-top: 2px solid #c8922a;
    border-radius: 3px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #556070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    color: #e8a832;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    background: #c8922a;
    color: #080c10;
    border: none;
    border-radius: 2px;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.55rem 1.8rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #e8a832; }

/* Arabic text area */
.stTextArea textarea {
    font-family: 'Amiri', serif !important;
    font-size: 1.2rem !important;
    line-height: 2.0 !important;
    background: #0a0f15 !important;
    color: #ddd4c0 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 2px !important;
    direction: rtl !important;
    text-align: right !important;
}

.stSelectbox > div > div, .stTextInput > div > div > input {
    background: #0a0f15 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #ddd4c0 !important;
}

hr { border-color: #1e2530; }

/* Entity chips */
.tag-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 3px 4px;
    vertical-align: middle;
    white-space: nowrap;
}
.tag-PER  { background: #2c1a4a; color: #c4a0ff; border: 1px solid #6040aa; }
.tag-LOC  { background: #0a2e20; color: #50e8a0; border: 1px solid #1a8855; }
.tag-ORG  { background: #2e1800; color: #ffaa50; border: 1px solid #aa5510; }
.tag-MISC { background: #1e2020; color: #9aaa9a; border: 1px solid #3a4040; }

/* RTL annotated output block */
.token-block {
    direction: rtl;
    text-align: right;
    line-height: 3.0;
    font-family: 'Amiri', serif;
    font-size: 1.3rem;
    background: #0a0f15;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 20px 24px;
    min-height: 90px;
}

/* Status badges */
.status-ok   { color: #50e8a0; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }
.status-err  { color: #ff6b6b; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }
.status-warn { color: #c8922a; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }

/* Info cards */
.info-card {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-left: 3px solid #c8922a;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 0 4px 4px 0;
}
.info-card h4 { font-family:'Syne',sans-serif; font-size:0.88rem; margin:0 0 4px; color:#c8922a; }
.info-card p  { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#556070; margin:0; }

/* Classification result card */
.cls-card {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 20px 24px;
    text-align: center;
}

.stProgress > div > div { background: #c8922a !important; }

button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* JSON expander */
.stExpander { border: 1px solid #1e2530 !important; border-radius: 3px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Entity colour map (WikiANN labels from backend)
# ─────────────────────────────────────────────
ENTITY_META = {
    "PER":  ("Person",        "tag-PER"),
    "LOC":  ("Location",      "tag-LOC"),
    "ORG":  ("Organization",  "tag-ORG"),
    "MISC": ("Miscellaneous", "tag-MISC"),
}

# SIB-200 topic labels returned by /api/classification
TOPIC_COLORS = {
    "science/technology": "#7c6af7",
    "travel":             "#00d4aa",
    "politics":           "#f97316",
    "sports":             "#3b82f6",
    "health":             "#ec4899",
    "entertainment":      "#fbbf24",
    "geography":          "#10b981",
}


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────
def _label_root(label: str) -> str:
    """Strip B-/I- prefix → PER / LOC / ORG / MISC."""
    return label.replace("B-", "").replace("I-", "").upper()


def call_ner(text: str, base_url: str, timeout: int) -> dict:
    resp = requests.post(
        f"{base_url}/api/ner",
        json={"text": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def call_classification(text: str, base_url: str, timeout: int) -> dict:
    resp = requests.post(
        f"{base_url}/api/classification",
        json={"text": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def call_health(base_url: str, timeout: int = 5) -> dict:
    resp = requests.get(f"{base_url}/health", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_ner_batch(texts: list, base_url: str, timeout: int) -> list:
    resp = requests.post(
        f"{base_url}/api/ner/batch",
        json={"texts": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────
def render_ner_annotated(text: str, entities: list) -> str:
    """
    Reconstruct annotated text from the character-level span offsets
    returned by the backend (entity.start / entity.end).
    Falls back to word-level highlighting if spans are missing.
    """
    if not entities:
        return f'<div class="token-block">{text}</div>'

    # Build span list sorted by start offset
    spans = sorted(entities, key=lambda e: e.get("start", 0))
    result = ""
    cursor = 0

    for ent in spans:
        start = ent.get("start", 0)
        end   = ent.get("end", 0)
        word  = ent.get("word", "")
        root  = _label_root(ent.get("entity_group", "MISC"))
        meta  = ENTITY_META.get(root, ("Unknown", "tag-MISC"))
        css   = meta[1]
        score = ent.get("score", 0)

        # Plain text before this entity
        if start > cursor:
            result += text[cursor:start]

        result += (
            f'<span class="tag-chip {css}">'
            f'<span style="font-family:Amiri,serif;font-size:1.1rem">{word}</span>'
            f'&nbsp;<b>{root}</b>'
            f'<span style="font-size:0.58rem;opacity:0.5;margin-left:4px">{score:.2f}</span>'
            f'</span>'
        )
        cursor = end

    # Remaining text after last entity
    if cursor < len(text):
        result += text[cursor:]

    return f'<div class="token-block">{result}</div>'


def render_entity_table(entities: list) -> str:
    rows = ""
    for ent in entities:
        root = _label_root(ent.get("entity_group", "MISC"))
        meta = ENTITY_META.get(root, ("Unknown", "tag-MISC"))
        css  = meta[1]
        name = meta[0]
        word  = ent.get("word", "")
        score = ent.get("score", 0)
        rows += (
            f"<tr>"
            f"<td style='padding:7px 14px;text-align:right;direction:rtl;"
            f"font-family:Amiri,serif;font-size:1.1rem;color:#ddd4c0'>{word}</td>"
            f"<td style='padding:7px 14px'><span class='tag-chip {css}'>{root}</span></td>"
            f"<td style='padding:7px 14px;font-family:IBM Plex Mono,monospace;"
            f"font-size:0.7rem;color:#556070'>{name}</td>"
            f"<td style='padding:7px 14px;font-family:IBM Plex Mono,monospace;"
            f"font-size:0.73rem;color:#c8922a'>{score:.4f}</td>"
            f"</tr>"
        )
    header = (
        "<table style='width:100%;border-collapse:collapse;border:1px solid #1e2530'>"
        "<thead><tr>"
        + "".join(
            f"<th style='text-align:left;font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
            f"color:#3a5070;padding:6px 14px;border-bottom:1px solid #1e2530;background:#0d1117'>{h}</th>"
            for h in ["Token", "Label", "Type", "Confidence"]
        )
        + f"</tr></thead><tbody>{rows}</tbody></table>"
    )
    return header


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-size:1.1rem;color:#c8922a">⚙ Backend Config</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="FastAPI backend URL (e.g. http://localhost:8000)",
    )
    request_timeout = st.slider("Request timeout (s)", 5, 120, 30)

    st.markdown("---")

    # Live health check
    if st.button("🔍 Check Backend Health"):
        try:
            health = call_health(api_url, timeout=5)
            st.markdown(
                f'<p class="status-ok">● Backend online — v{health.get("version","?")}</p>',
                unsafe_allow_html=True,
            )
            loaded = health.get("models_loaded", [])
            if loaded:
                st.markdown(
                    f'<p class="status-ok">✓ Models loaded: {", ".join(loaded)}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="status-warn">⚠ No models loaded yet (lazy loading — run an inference first)</p>',
                    unsafe_allow_html=True,
                )
            paths = health.get("model_paths", {})
            for task, path in paths.items():
                short = path if len(path) < 50 else "…" + path[-46:]
                st.markdown(
                    f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#3a5070">'
                    f'{task}: {short}</p>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.markdown(
                f'<p class="status-err">✗ Cannot reach backend: {e}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Entity filter (for NER tab display only)
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#556070">Entity Filters (display only)</p>',
        unsafe_allow_html=True,
    )
    ent_toggles = {code: st.checkbox(f"{code} — {name}", value=True)
                   for code, (name, _) in ENTITY_META.items()}

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.65rem;color:#2a3540;font-family:IBM Plex Mono,monospace">'
        'Arabic NLP Pipeline v1.0 · FastAPI + XLM-R</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(
        '<h1 style="font-size:2.6rem;font-weight:800;margin-bottom:2px">Arabic NLP Pipeline</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.76rem;color:#3a5070;margin:0 0 20px">'
        f'Low-Resource NER + Text Classification &nbsp;·&nbsp; XLM-RoBERTa &nbsp;·&nbsp; WikiANN / SIB-200'
        f'</p>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1e2530;border-top:2px solid #c8922a;'
        'padding:10px 14px;text-align:center;border-radius:3px;margin-top:8px">'
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#556070;margin:0">LANGUAGE</p>'
        '<p style="font-family:\'Amiri\',serif;font-size:1.6rem;color:#c8922a;margin:4px 0 0">عربي</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# Entity legend
legend_html = '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 20px;align-items:center">'
for code, (name, css) in ENTITY_META.items():
    legend_html += (
        f'<div style="display:flex;align-items:center;gap:6px;font-family:IBM Plex Mono,monospace;'
        f'font-size:0.7rem;color:#6a7a8a"><span class="tag-chip {css}" style="margin:0">{code}</span>{name}</div>'
    )
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_ner, tab_cls, tab_batch, tab_api = st.tabs([
    "◈  NER",
    "◉  CLASSIFICATION",
    "◎  BATCH NER",
    "⊕  API INFO",
])

# ══════════════════════════════════════════════
# TAB 1 — NER
# ══════════════════════════════════════════════
with tab_ner:
    SAMPLES_NER = {
        "Sample 1 — People & Places":    "سافر محمد إلى القاهرة في يناير ٢٠٢٤ لحضور مؤتمر اليونسكو.",
        "Sample 2 — Politics & History": "وقّعت فاطمة وأحمد اتفاقية في الرياض بحضور ممثلي الأمم المتحدة.",
        "Sample 3 — Education":          "تدرس نورا اللغة العربية في جامعة بيروت منذ مارس 2024.",
        "Sample 4 — News":               "أعلنت وزارة الصحة السعودية عن اكتشاف علاج جديد للسرطان في الرياض.",
        "Custom (blank)":                "",
    }

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a;margin-bottom:6px">Input Text</p>', unsafe_allow_html=True)
        sample_choice = st.selectbox("Load sample", list(SAMPLES_NER.keys()), key="ner_sample")
        user_text_ner = st.text_area(
            "Arabic text for NER",
            value=SAMPLES_NER[sample_choice],
            height=160,
            placeholder="Type or paste Arabic text here…",
            label_visibility="collapsed",
            key="ner_input",
        )
        run_ner = st.button("⚡  Run NER", key="run_ner")

    with col_out:
        st.markdown('<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a;margin-bottom:6px">Annotated Output</p>', unsafe_allow_html=True)

        if run_ner:
            if not user_text_ner.strip():
                st.warning("Please enter some Arabic text first.")
            else:
                with st.spinner("Calling /api/ner…"):
                    try:
                        t0 = time.perf_counter()
                        result = call_ner(user_text_ner, api_url, request_timeout)
                        elapsed = (time.perf_counter() - t0) * 1000

                        entities = result.get("entities", [])
                        # Apply display filter
                        visible = [e for e in entities
                                   if ent_toggles.get(_label_root(e.get("entity_group", "")), True)]

                        # Annotated text
                        st.markdown(
                            render_ner_annotated(user_text_ner, visible),
                            unsafe_allow_html=True,
                        )

                        # Stats row
                        st.markdown("")
                        latency_server = result.get("processing_time_ms", 0)
                        model_path     = result.get("model_path", "—")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Entities Found", len(entities))
                        c2.metric("Server Latency", f"{latency_server:.0f} ms")
                        c3.metric("Round-trip",     f"{elapsed:.0f} ms")

                        # Entity table
                        if visible:
                            st.markdown("")
                            st.markdown(
                                '<p style="font-family:\'Syne\',sans-serif;font-size:0.82rem;color:#c8922a">Detected Entities</p>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(render_entity_table(visible), unsafe_allow_html=True)

                            # Count badges
                            counts = Counter(_label_root(e.get("entity_group", "")) for e in visible)
                            st.markdown("")
                            count_html = '<div style="display:flex;gap:10px;flex-wrap:wrap">'
                            for code, cnt in counts.items():
                                css = ENTITY_META.get(code, ("", "tag-MISC"))[1]
                                count_html += f'<span class="tag-chip {css}">{code} <b style="font-size:1rem">{cnt}</b></span>'
                            count_html += "</div>"
                            st.markdown(count_html, unsafe_allow_html=True)
                        else:
                            st.info("No entities detected (or all filtered out).")

                        # Model path footnote
                        st.markdown(
                            f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#2a3a4a;margin-top:10px">'
                            f'Model: {model_path}</p>',
                            unsafe_allow_html=True,
                        )

                        # Raw JSON expander
                        with st.expander("Raw API response"):
                            st.json(result)

                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot connect to backend at **{api_url}**. Is the FastAPI server running?")
                    except requests.exceptions.Timeout:
                        st.error(f"Request timed out after {request_timeout}s.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"Backend returned HTTP {e.response.status_code}: {e.response.text}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        else:
            st.markdown(
                '<div class="token-block" style="color:#2a3540;font-size:0.85rem;font-family:IBM Plex Mono,monospace">'
                '← Enter Arabic text and click Run NER</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════
with tab_cls:
    SAMPLES_CLS = {
        "Sample 1 — Technology":   "أحدث الهواتف الذكية في السوق تتميز بكاميرات متطورة وذكاء اصطناعي.",
        "Sample 2 — Sports":       "فاز الفريق الوطني ببطولة العالم لكرة القدم بعد مباراة مثيرة.",
        "Sample 3 — Health":       "أعلن الأطباء عن علاج جديد لمرض السكري يعتمد على الخلايا الجذعية.",
        "Sample 4 — Politics":     "عقد البرلمان جلسة طارئة لمناقشة الميزانية العامة للدولة.",
        "Sample 5 — Travel":       "يعدّ السفر إلى المغرب تجربة ثقافية فريدة بمناظره الطبيعية الخلابة.",
        "Sample 6 — Entertainment":"حقق الفيلم العربي الجديد أعلى الإيرادات في تاريخ السينما العربية.",
        "Custom (blank)":          "",
    }

    col_cls_in, col_cls_out = st.columns([1, 1], gap="large")

    with col_cls_in:
        st.markdown('<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a;margin-bottom:6px">Input Text</p>', unsafe_allow_html=True)
        cls_sample = st.selectbox("Load sample", list(SAMPLES_CLS.keys()), key="cls_sample")
        user_text_cls = st.text_area(
            "Arabic text for classification",
            value=SAMPLES_CLS[cls_sample],
            height=160,
            placeholder="Type or paste Arabic text here…",
            label_visibility="collapsed",
            key="cls_input",
        )
        run_cls = st.button("⚡  Run Classification", key="run_cls")

        st.markdown("")
        st.markdown(
            '<div class="info-card">'
            '<h4>Topic Labels (SIB-200)</h4>'
            '<p>science/technology · travel · politics<br>'
            'sports · health · entertainment · geography</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_cls_out:
        st.markdown('<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a;margin-bottom:6px">Classification Result</p>', unsafe_allow_html=True)

        if run_cls:
            if not user_text_cls.strip():
                st.warning("Please enter some Arabic text first.")
            else:
                with st.spinner("Calling /api/classification…"):
                    try:
                        t0 = time.perf_counter()
                        result = call_classification(user_text_cls, api_url, request_timeout)
                        elapsed = (time.perf_counter() - t0) * 1000

                        label      = result.get("label", "unknown")
                        score      = result.get("score", 0)
                        latency_sv = result.get("processing_time_ms", 0)
                        model_path = result.get("model_path", "—")
                        color      = TOPIC_COLORS.get(label.lower(), "#c8922a")

                        # Big result card
                        st.markdown(
                            f'<div class="cls-card">'
                            f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#556070;margin:0 0 8px">PREDICTED TOPIC</p>'
                            f'<p style="font-family:\'Syne\',sans-serif;font-size:2.2rem;font-weight:800;color:{color};margin:0 0 6px">{label.upper()}</p>'
                            f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#ddd4c0;margin:0">Confidence: <b style="color:{color}">{score*100:.1f}%</b></p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Confidence bar
                        st.markdown("")
                        st.progress(score)

                        # Stats
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence",    f"{score*100:.1f}%")
                        c2.metric("Server Latency", f"{latency_sv:.0f} ms")
                        c3.metric("Round-trip",     f"{elapsed:.0f} ms")

                        # Model path
                        st.markdown(
                            f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#2a3a4a;margin-top:6px">'
                            f'Model: {model_path}</p>',
                            unsafe_allow_html=True,
                        )

                        with st.expander("Raw API response"):
                            st.json(result)

                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot connect to backend at **{api_url}**. Is the FastAPI server running?")
                    except requests.exceptions.Timeout:
                        st.error(f"Request timed out after {request_timeout}s.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"Backend returned HTTP {e.response.status_code}: {e.response.text}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        else:
            st.markdown(
                '<div style="background:#0a0f15;border:1px solid #1e2530;border-radius:4px;'
                'padding:40px 24px;text-align:center;color:#2a3540;font-family:IBM Plex Mono,monospace;font-size:0.85rem">'
                '← Enter Arabic text and click Run Classification</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TAB 3 — BATCH NER
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.76rem;color:#556070">'
        'Submit up to 50 Arabic sentences at once via <b style="color:#c8922a">POST /api/ner/batch</b></p>',
        unsafe_allow_html=True,
    )

    default_batch = (
        "سافر محمد إلى القاهرة في يناير ٢٠٢٤.\n"
        "أعلنت وزارة الصحة السعودية عن إجراءات جديدة.\n"
        "فازت مصر ببطولة أفريقيا لكرة القدم في الرياض.\n"
        "تأسست منظمة اليونسكو في باريس عام ١٩٤٥."
    )

    batch_text = st.text_area(
        "One Arabic sentence per line (max 50)",
        value=default_batch,
        height=200,
        key="batch_input",
    )

    run_batch = st.button("⚡  Run Batch NER", key="run_batch")

    if run_batch:
        lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one sentence.")
        elif len(lines) > 50:
            st.error("Maximum 50 sentences per batch.")
        else:
            with st.spinner(f"Calling /api/ner/batch with {len(lines)} sentences…"):
                try:
                    t0 = time.perf_counter()
                    results = call_ner_batch(lines, api_url, request_timeout)
                    elapsed = (time.perf_counter() - t0) * 1000

                    st.markdown(
                        f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#556070">'
                        f'Processed {len(results)} sentences in {elapsed:.0f} ms</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

                    all_entity_counts: Counter = Counter()

                    for i, item in enumerate(results):
                        txt      = item.get("text", "")
                        ents     = item.get("entities", [])
                        visible  = [e for e in ents if ent_toggles.get(_label_root(e.get("entity_group", "")), True)]

                        # Per-sentence expander
                        with st.expander(f"Sentence {i+1}  —  {len(ents)} entities found", expanded=(i == 0)):
                            st.markdown(
                                f'<div style="font-family:Amiri,serif;font-size:1.15rem;direction:rtl;'
                                f'text-align:right;background:#0a0f15;border:1px solid #1e2530;'
                                f'border-radius:3px;padding:12px 16px;margin-bottom:10px">{txt}</div>',
                                unsafe_allow_html=True,
                            )
                            if visible:
                                st.markdown(render_entity_table(visible), unsafe_allow_html=True)
                                for e in visible:
                                    all_entity_counts[_label_root(e.get("entity_group", ""))] += 1
                            else:
                                st.markdown(
                                    '<p style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#3a5070">No entities detected.</p>',
                                    unsafe_allow_html=True,
                                )

                    # Aggregate summary
                    st.markdown("---")
                    st.markdown(
                        '<p style="font-family:\'Syne\',sans-serif;font-size:0.9rem;color:#c8922a">Batch Summary</p>',
                        unsafe_allow_html=True,
                    )
                    total_ents = sum(all_entity_counts.values())
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Sentences",       len(results))
                    b2.metric("Total Entities",  total_ents)
                    b3.metric("Avg / Sentence",  f"{total_ents/max(len(results),1):.1f}")

                    if all_entity_counts:
                        count_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
                        for code, cnt in all_entity_counts.most_common():
                            css = ENTITY_META.get(code, ("", "tag-MISC"))[1]
                            count_html += f'<span class="tag-chip {css}">{code} <b style="font-size:1rem">{cnt}</b></span>'
                        count_html += "</div>"
                        st.markdown(count_html, unsafe_allow_html=True)

                    with st.expander("Full raw JSON response"):
                        st.json(results)

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at **{api_url}**.")
                except requests.exceptions.Timeout:
                    st.error(f"Request timed out after {request_timeout}s.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Backend returned HTTP {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ══════════════════════════════════════════════
# TAB 4 — API INFO
# ══════════════════════════════════════════════
with tab_api:
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.76rem;color:#556070">'
        'Reference for the FastAPI backend endpoints consumed by this UI</p>',
        unsafe_allow_html=True,
    )

    endpoints = [
        ("GET",  "/health",              "Health check — returns backend status, loaded models, model paths"),
        ("POST", "/api/ner",             "Named Entity Recognition on a single Arabic text (WikiANN: PER, ORG, LOC)"),
        ("POST", "/api/ner/batch",       "Batch NER — up to 50 texts in one request"),
        ("POST", "/api/classification",  "Text topic classification (SIB-200: 7 categories)"),
    ]

    for method, path, desc in endpoints:
        color = "#50e8a0" if method == "GET" else "#c8922a"
        st.markdown(
            f'<div class="info-card">'
            f'<h4><span style="color:{color}">{method}</span> &nbsp; {path}</h4>'
            f'<p>{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_req, col_res = st.columns(2, gap="large")

    with col_req:
        st.markdown(
            '<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a">Request Schema</p>',
            unsafe_allow_html=True,
        )
        st.code(
            '// POST /api/ner  or  /api/classification\n'
            '{\n'
            '  "text": "Arabic text string (1–2000 chars)"\n'
            '}\n\n'
            '// POST /api/ner/batch\n'
            '{\n'
            '  "texts": ["sentence 1", "sentence 2", ...]\n'
            '}',
            language="json",
        )

    with col_res:
        st.markdown(
            '<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a">Response Examples</p>',
            unsafe_allow_html=True,
        )
        st.code(
            '// NER response\n'
            '{\n'
            '  "text": "...",\n'
            '  "entities": [\n'
            '    { "entity_group": "PER",\n'
            '      "word": "محمد",\n'
            '      "score": 0.9921,\n'
            '      "start": 6, "end": 9 }\n'
            '  ],\n'
            '  "model_path": "outputs/ner_model",\n'
            '  "processing_time_ms": 38.4\n'
            '}\n\n'
            '// Classification response\n'
            '{\n'
            '  "text": "...",\n'
            '  "label": "sports",\n'
            '  "score": 0.9734,\n'
            '  "model_path": "outputs/classification_model",\n'
            '  "processing_time_ms": 22.1\n'
            '}',
            language="json",
        )

    st.markdown("---")

    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-size:0.85rem;color:#c8922a">Model Architecture</p>',
        unsafe_allow_html=True,
    )
    arch_items = [
        ("#c8922a", "#1a1200", "#3a2800", "► Arabic Input Tokens",        "(RTL BPE / SentencePiece)"),
        ("#5050c0", "#0d0d20", "#1a1a40", "Multilingual Embedding Layer", "(xlm-roberta-base, 250k vocab)"),
        ("#20a860", "#0a1a10", "#1a3a20", "Transformer Encoder × 12",     "(cross-lingual pretrained weights)"),
        ("#2080a0", "#0a1520", "#1a3050", "Task Head (NER)",               "(Token classification → BIO labels)"),
        ("#905020", "#1a1008", "#3a2010", "Task Head (Classification)",    "(Sequence classification → 7 topics)"),
        ("#c8922a", "#1a1200", "#3a2800", "► Predictions + Confidence",   ""),
    ]
    arch_col, _ = st.columns([2, 3])
    with arch_col:
        arch_html = "<div>"
        for i, (color, bg, border_c, lbl, sub) in enumerate(arch_items):
            arrow = '<div style="text-align:center;color:#222;font-size:0.9rem;margin:2px 0">↓</div>' if i < len(arch_items) - 1 else ""
            arch_html += (
                f'<div style="background:{bg};border:1px solid {border_c};padding:9px 14px;border-radius:3px;margin-bottom:2px">'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:{color}">{lbl}</span>'
                + (f'<br><span style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#334">{sub}</span>' if sub else "")
                + f'</div>{arrow}'
            )
        arch_html += "</div>"
        st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #1e2530;border-left:3px solid #556070;'
        f'padding:12px 18px;border-radius:0 3px 3px 0">'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#556070">'
        f'ℹ Start the backend with: <b style="color:#c8922a">uvicorn main:app --host 0.0.0.0 --port 8000</b>'
        f'<br>Interactive Swagger docs available at: <b style="color:#c8922a">{api_url}/docs</b>'
        f'</span></div>',
        unsafe_allow_html=True,
    )
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
# CSS — Warm Editorial / Human-Crafted Aesthetic
# Inspired by: research tools, typesetting, print design
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+3:wght@300;400;600&family=Amiri:wght@400;700&display=swap');

/* ── Core ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #f5f0e8;
    color: #2c2416;
}
h1,h2,h3,h4 {
    font-family: 'Playfair Display', serif;
    letter-spacing: -0.01em;
    color: #1a1208;
}

/* ── Sidebar ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #ede8df;
    border-right: 1px solid #c8bca8;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.8rem !important;
    color: #7a6a54 !important;
}

/* ── Metric Cards ─────────────────────────── */
div[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #d8cebb;
    border-bottom: 3px solid #8b5e3c;
    border-radius: 2px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(44,36,22,0.06);
}
div[data-testid="metric-container"] label {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    color: #9a8a74;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: #2c2416;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    font-family: 'Source Sans 3', sans-serif;
    background: #8b5e3c;
    color: #faf7f2;
    border: none;
    border-radius: 2px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.8rem;
    transition: background 0.2s;
    box-shadow: 0 1px 4px rgba(139,94,60,0.25);
}
.stButton > button:hover { background: #6e4a2e; }

/* ── Arabic Text Area ─────────────────────── */
.stTextArea textarea {
    font-family: 'Amiri', serif !important;
    font-size: 1.2rem !important;
    line-height: 2.0 !important;
    background: #fffdf9 !important;
    color: #2c2416 !important;
    border: 1px solid #c8bca8 !important;
    border-radius: 2px !important;
    direction: rtl !important;
    text-align: right !important;
    box-shadow: inset 0 1px 3px rgba(44,36,22,0.05) !important;
}

.stSelectbox > div > div, .stTextInput > div > div > input {
    background: #fffdf9 !important;
    border: 1px solid #c8bca8 !important;
    border-radius: 2px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.85rem !important;
    color: #2c2416 !important;
}

hr { border-color: #d8cebb; }

/* ── Entity Chips ─────────────────────────── */
.tag-chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 2px;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 3px 4px;
    vertical-align: middle;
    white-space: nowrap;
}
/* Muted ink-on-paper palette */
.tag-PER  { background: #e8e0f0; color: #4a2e7a; border: 1px solid #b8a0d8; }
.tag-LOC  { background: #ddf0e8; color: #1a5c38; border: 1px solid #88c8a8; }
.tag-ORG  { background: #f0e8d8; color: #7a4010; border: 1px solid #c8a870; }
.tag-MISC { background: #e8e8e4; color: #4a4a40; border: 1px solid #b0b0a0; }

/* ── RTL Annotated Block ──────────────────── */
.token-block {
    direction: rtl;
    text-align: right;
    line-height: 3.0;
    font-family: 'Amiri', serif;
    font-size: 1.3rem;
    background: #fffdf9;
    border: 1px solid #c8bca8;
    border-radius: 2px;
    padding: 20px 24px;
    min-height: 90px;
    box-shadow: inset 0 1px 4px rgba(44,36,22,0.04);
}

/* ── Status Badges ────────────────────────── */
.status-ok   { color: #2a6e44; font-family: 'Source Sans 3', sans-serif; font-size: 0.78rem; font-weight: 600; }
.status-err  { color: #8b2222; font-family: 'Source Sans 3', sans-serif; font-size: 0.78rem; font-weight: 600; }
.status-warn { color: #7a5010; font-family: 'Source Sans 3', sans-serif; font-size: 0.78rem; font-weight: 600; }

/* ── Info Cards ───────────────────────────── */
.info-card {
    background: #fffdf9;
    border: 1px solid #d8cebb;
    border-left: 3px solid #8b5e3c;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 0 2px 2px 0;
    box-shadow: 0 1px 3px rgba(44,36,22,0.05);
}
.info-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    margin: 0 0 4px;
    color: #2c2416;
}
.info-card p {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.78rem;
    color: #7a6a54;
    margin: 0;
}

/* ── Classification Result Card ───────────── */
.cls-card {
    background: #fffdf9;
    border: 1px solid #d8cebb;
    border-radius: 2px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(44,36,22,0.07);
}

/* ── Progress Bar ─────────────────────────── */
.stProgress > div > div { background: #8b5e3c !important; }

/* ── Tabs ─────────────────────────────────── */
button[data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #7a6a54 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2c2416 !important;
    border-bottom: 2px solid #8b5e3c !important;
}

/* ── Expander ─────────────────────────────── */
.stExpander {
    border: 1px solid #d8cebb !important;
    border-radius: 2px !important;
    background: #fffdf9 !important;
}

/* ── Subtle page texture ──────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(180,150,100,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at bottom right, rgba(100,80,50,0.03) 0%, transparent 60%),
        #f5f0e8;
}
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
    "science/technology": "#4a3a8a",
    "travel":             "#1a6a5a",
    "politics":           "#8a3a1a",
    "sports":             "#1a4a7a",
    "health":             "#7a2a4a",
    "entertainment":      "#6a5010",
    "geography":          "#1a5a3a",
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
    if not entities:
        return f'<div class="token-block">{text}</div>'

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

        if start > cursor:
            result += text[cursor:start]

        result += (
            f'<span class="tag-chip {css}">'
            f'<span style="font-family:Amiri,serif;font-size:1.1rem">{word}</span>'
            f'&nbsp;{root}'
            f'<span style="font-size:0.62rem;opacity:0.55;margin-left:4px">{score:.2f}</span>'
            f'</span>'
        )
        cursor = end

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
            f"<td style='padding:8px 14px;text-align:right;direction:rtl;"
            f"font-family:Amiri,serif;font-size:1.1rem;color:#2c2416;border-bottom:1px solid #ede8df'>{word}</td>"
            f"<td style='padding:8px 14px;border-bottom:1px solid #ede8df'><span class='tag-chip {css}'>{root}</span></td>"
            f"<td style='padding:8px 14px;font-family:Source Sans 3,sans-serif;"
            f"font-size:0.78rem;color:#9a8a74;border-bottom:1px solid #ede8df'>{name}</td>"
            f"<td style='padding:8px 14px;font-family:Source Sans 3,sans-serif;"
            f"font-size:0.78rem;color:#5a4a34;border-bottom:1px solid #ede8df'>{score:.4f}</td>"
            f"</tr>"
        )
    header = (
        "<table style='width:100%;border-collapse:collapse;border:1px solid #d8cebb;background:#fffdf9;border-radius:2px'>"
        "<thead><tr>"
        + "".join(
            f"<th style='text-align:left;font-family:Source Sans 3,sans-serif;font-size:0.7rem;"
            f"font-weight:600;text-transform:uppercase;letter-spacing:0.08em;"
            f"color:#9a8a74;padding:8px 14px;border-bottom:2px solid #d8cebb;background:#f5f0e8'>{h}</th>"
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
        '<h2 style="font-family:\'Playfair Display\',serif;font-size:1.15rem;color:#2c2416;margin-bottom:4px">Backend Config</h2>',
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

    if st.button("Check Backend Health"):
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
                    '<p class="status-warn">⚠ No models loaded yet (lazy loading)</p>',
                    unsafe_allow_html=True,
                )
            paths = health.get("model_paths", {})
            for task, path in paths.items():
                short = path if len(path) < 50 else "…" + path[-46:]
                st.markdown(
                    f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.7rem;color:#b0a090">'
                    f'{task}: {short}</p>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.markdown(
                f'<p class="status-err">✗ Cannot reach backend: {e}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown(
        '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:0.72rem;color:#9a8a74;text-transform:uppercase;letter-spacing:0.08em">Entity Filters</p>',
        unsafe_allow_html=True,
    )
    ent_toggles = {code: st.checkbox(f"{code} — {name}", value=True)
                   for code, (name, _) in ENTITY_META.items()}

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.68rem;color:#c0b0a0;font-family:Source Sans 3,sans-serif">'
        'Arabic NLP Pipeline v1.0<br>FastAPI + XLM-RoBERTa</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(
        '<h1 style="font-size:2.8rem;font-weight:900;margin-bottom:2px;color:#1a1208">Arabic NLP Pipeline</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:0.88rem;color:#9a8a74;margin:0 0 20px;font-style:italic">'
        'Named Entity Recognition &amp; Text Classification &nbsp;·&nbsp; XLM-RoBERTa &nbsp;·&nbsp; WikiANN / SIB-200'
        '</p>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        '<div style="background:#fffdf9;border:1px solid #d8cebb;border-top:3px solid #8b5e3c;'
        'padding:10px 14px;text-align:center;border-radius:2px;margin-top:8px;box-shadow:0 1px 4px rgba(44,36,22,0.08)">'
        '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:0.6rem;color:#9a8a74;margin:0;text-transform:uppercase;letter-spacing:0.1em">Language</p>'
        '<p style="font-family:\'Amiri\',serif;font-size:1.6rem;color:#2c2416;margin:4px 0 0">عربي</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# Entity legend
legend_html = '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 20px;align-items:center">'
for code, (name, css) in ENTITY_META.items():
    legend_html += (
        f'<div style="display:flex;align-items:center;gap:6px;font-family:Source Sans 3,sans-serif;'
        f'font-size:0.75rem;color:#9a8a74"><span class="tag-chip {css}" style="margin:0">{code}</span>{name}</div>'
    )
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_ner, tab_cls, tab_batch, tab_api = st.tabs([
    "NER",
    "Classification",
    "Batch NER",
    "API Info",
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
        st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416;margin-bottom:6px">Input Text</p>', unsafe_allow_html=True)
        sample_choice = st.selectbox("Load sample", list(SAMPLES_NER.keys()), key="ner_sample")
        user_text_ner = st.text_area(
            "Arabic text for NER",
            value=SAMPLES_NER[sample_choice],
            height=160,
            placeholder="Type or paste Arabic text here…",
            label_visibility="collapsed",
            key="ner_input",
        )
        run_ner = st.button("Run NER →", key="run_ner")

    with col_out:
        st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416;margin-bottom:6px">Annotated Output</p>', unsafe_allow_html=True)

        if run_ner:
            if not user_text_ner.strip():
                st.warning("Please enter some Arabic text first.")
            else:
                with st.spinner("Running entity recognition…"):
                    try:
                        t0 = time.perf_counter()
                        result = call_ner(user_text_ner, api_url, request_timeout)
                        elapsed = (time.perf_counter() - t0) * 1000

                        entities = result.get("entities", [])
                        visible = [e for e in entities
                                   if ent_toggles.get(_label_root(e.get("entity_group", "")), True)]

                        st.markdown(render_ner_annotated(user_text_ner, visible), unsafe_allow_html=True)

                        st.markdown("")
                        latency_server = result.get("processing_time_ms", 0)
                        model_path     = result.get("model_path", "—")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Entities Found", len(entities))
                        c2.metric("Server Latency", f"{latency_server:.0f} ms")
                        c3.metric("Round-trip",     f"{elapsed:.0f} ms")

                        if visible:
                            st.markdown("")
                            st.markdown(
                                '<p style="font-family:\'Playfair Display\',serif;font-size:0.9rem;color:#2c2416">Detected Entities</p>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(render_entity_table(visible), unsafe_allow_html=True)

                            counts = Counter(_label_root(e.get("entity_group", "")) for e in visible)
                            st.markdown("")
                            count_html = '<div style="display:flex;gap:10px;flex-wrap:wrap">'
                            for code, cnt in counts.items():
                                css = ENTITY_META.get(code, ("", "tag-MISC"))[1]
                                count_html += f'<span class="tag-chip {css}">{code} {cnt}</span>'
                            count_html += "</div>"
                            st.markdown(count_html, unsafe_allow_html=True)
                        else:
                            st.info("No entities detected (or all filtered out).")

                        st.markdown(
                            f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.68rem;color:#c0b0a0;margin-top:10px">'
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
                '<div class="token-block" style="color:#c0b0a0;font-size:0.88rem;font-family:Source Sans 3,sans-serif;font-style:italic">'
                'Enter Arabic text and click Run NER</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════
with tab_cls:
    SAMPLES_CLS = {
        "Sample 1 — Technology":    "أحدث الهواتف الذكية في السوق تتميز بكاميرات متطورة وذكاء اصطناعي.",
        "Sample 2 — Sports":        "فاز الفريق الوطني ببطولة العالم لكرة القدم بعد مباراة مثيرة.",
        "Sample 3 — Health":        "أعلن الأطباء عن علاج جديد لمرض السكري يعتمد على الخلايا الجذعية.",
        "Sample 4 — Politics":      "عقد البرلمان جلسة طارئة لمناقشة الميزانية العامة للدولة.",
        "Sample 5 — Travel":        "يعدّ السفر إلى المغرب تجربة ثقافية فريدة بمناظره الطبيعية الخلابة.",
        "Sample 6 — Entertainment": "حقق الفيلم العربي الجديد أعلى الإيرادات في تاريخ السينما العربية.",
        "Custom (blank)":           "",
    }

    col_cls_in, col_cls_out = st.columns([1, 1], gap="large")

    with col_cls_in:
        st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416;margin-bottom:6px">Input Text</p>', unsafe_allow_html=True)
        cls_sample = st.selectbox("Load sample", list(SAMPLES_CLS.keys()), key="cls_sample")
        user_text_cls = st.text_area(
            "Arabic text for classification",
            value=SAMPLES_CLS[cls_sample],
            height=160,
            placeholder="Type or paste Arabic text here…",
            label_visibility="collapsed",
            key="cls_input",
        )
        run_cls = st.button("Run Classification →", key="run_cls")

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
        st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416;margin-bottom:6px">Classification Result</p>', unsafe_allow_html=True)

        if run_cls:
            if not user_text_cls.strip():
                st.warning("Please enter some Arabic text first.")
            else:
                with st.spinner("Classifying text…"):
                    try:
                        t0 = time.perf_counter()
                        result = call_classification(user_text_cls, api_url, request_timeout)
                        elapsed = (time.perf_counter() - t0) * 1000

                        label      = result.get("label", "unknown")
                        score      = result.get("score", 0)
                        latency_sv = result.get("processing_time_ms", 0)
                        model_path = result.get("model_path", "—")
                        color      = TOPIC_COLORS.get(label.lower(), "#5a4030")

                        st.markdown(
                            f'<div class="cls-card">'
                            f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.7rem;color:#9a8a74;margin:0 0 10px;text-transform:uppercase;letter-spacing:0.1em">Predicted Topic</p>'
                            f'<p style="font-family:\'Playfair Display\',serif;font-size:2.4rem;font-weight:700;color:{color};margin:0 0 8px">{label.upper()}</p>'
                            f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.88rem;color:#7a6a54;margin:0">Confidence: <b style="color:{color}">{score*100:.1f}%</b></p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown("")
                        st.progress(score)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence",     f"{score*100:.1f}%")
                        c2.metric("Server Latency", f"{latency_sv:.0f} ms")
                        c3.metric("Round-trip",     f"{elapsed:.0f} ms")

                        st.markdown(
                            f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.68rem;color:#c0b0a0;margin-top:6px">'
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
                '<div style="background:#fffdf9;border:1px solid #d8cebb;border-radius:2px;'
                'padding:40px 24px;text-align:center;color:#c0b0a0;font-family:Source Sans 3,sans-serif;font-size:0.88rem;font-style:italic">'
                'Enter Arabic text and click Run Classification</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TAB 3 — BATCH NER
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown(
        '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:0.82rem;color:#9a8a74">'
        'Submit up to 50 Arabic sentences at once via <b style="color:#5a4030">POST /api/ner/batch</b></p>',
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

    run_batch = st.button("Run Batch NER →", key="run_batch")

    if run_batch:
        lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one sentence.")
        elif len(lines) > 50:
            st.error("Maximum 50 sentences per batch.")
        else:
            with st.spinner(f"Processing {len(lines)} sentences…"):
                try:
                    t0 = time.perf_counter()
                    results = call_ner_batch(lines, api_url, request_timeout)
                    elapsed = (time.perf_counter() - t0) * 1000

                    st.markdown(
                        f'<p style="font-family:Source Sans 3,sans-serif;font-size:0.78rem;color:#9a8a74">'
                        f'Processed {len(results)} sentences in {elapsed:.0f} ms</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

                    all_entity_counts: Counter = Counter()

                    for i, item in enumerate(results):
                        txt     = item.get("text", "")
                        ents    = item.get("entities", [])
                        visible = [e for e in ents if ent_toggles.get(_label_root(e.get("entity_group", "")), True)]

                        with st.expander(f"Sentence {i+1}  —  {len(ents)} entities", expanded=(i == 0)):
                            st.markdown(
                                f'<div style="font-family:Amiri,serif;font-size:1.15rem;direction:rtl;'
                                f'text-align:right;background:#fffdf9;border:1px solid #d8cebb;'
                                f'border-radius:2px;padding:12px 16px;margin-bottom:10px">{txt}</div>',
                                unsafe_allow_html=True,
                            )
                            if visible:
                                st.markdown(render_entity_table(visible), unsafe_allow_html=True)
                                for e in visible:
                                    all_entity_counts[_label_root(e.get("entity_group", ""))] += 1
                            else:
                                st.markdown(
                                    '<p style="font-family:Source Sans 3,sans-serif;font-size:0.78rem;color:#c0b0a0;font-style:italic">No entities detected.</p>',
                                    unsafe_allow_html=True,
                                )

                    st.markdown("---")
                    st.markdown(
                        '<p style="font-family:\'Playfair Display\',serif;font-size:1rem;color:#2c2416">Batch Summary</p>',
                        unsafe_allow_html=True,
                    )
                    total_ents = sum(all_entity_counts.values())
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Sentences",      len(results))
                    b2.metric("Total Entities", total_ents)
                    b3.metric("Avg / Sentence", f"{total_ents/max(len(results),1):.1f}")

                    if all_entity_counts:
                        count_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
                        for code, cnt in all_entity_counts.most_common():
                            css = ENTITY_META.get(code, ("", "tag-MISC"))[1]
                            count_html += f'<span class="tag-chip {css}">{code} {cnt}</span>'
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
        '<p style="font-family:\'Source Sans 3\',sans-serif;font-size:0.82rem;color:#9a8a74;font-style:italic">'
        'Reference for the FastAPI backend endpoints consumed by this UI</p>',
        unsafe_allow_html=True,
    )

    endpoints = [
        ("GET",  "/health",             "Health check — returns backend status, loaded models, model paths"),
        ("POST", "/api/ner",            "Named Entity Recognition on a single Arabic text (WikiANN: PER, ORG, LOC)"),
        ("POST", "/api/ner/batch",      "Batch NER — up to 50 texts in one request"),
        ("POST", "/api/classification", "Text topic classification (SIB-200: 7 categories)"),
    ]

    for method, path, desc in endpoints:
        color = "#1a5c38" if method == "GET" else "#5a3010"
        st.markdown(
            f'<div class="info-card">'
            f'<h4><span style="color:{color};font-family:Source Sans 3,sans-serif;font-size:0.75rem;font-weight:700;letter-spacing:0.08em">{method}</span>'
            f'&nbsp;&nbsp;<span style="font-family:\'Playfair Display\',serif">{path}</span></h4>'
            f'<p>{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_req, col_res = st.columns(2, gap="large")

    with col_req:
        st.markdown(
            '<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416">Request Schema</p>',
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
            '<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416">Response Examples</p>',
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
        '<p style="font-family:\'Playfair Display\',serif;font-size:0.95rem;color:#2c2416">Model Architecture</p>',
        unsafe_allow_html=True,
    )
    arch_items = [
        ("#5a3010", "#f5ede0", "#d8c4a8", "Arabic Input Tokens",          "(RTL BPE / SentencePiece)"),
        ("#2a2a6a", "#eeeef8", "#b8b8d8", "Multilingual Embedding Layer", "(xlm-roberta-base, 250k vocab)"),
        ("#1a5c38", "#e4f4ec", "#90c8a8", "Transformer Encoder × 12",     "(cross-lingual pretrained weights)"),
        ("#1a3a6a", "#e4eef8", "#90b0d8", "Task Head (NER)",               "(Token classification → BIO labels)"),
        ("#6a2a10", "#f4e8e0", "#c8a090", "Task Head (Classification)",    "(Sequence classification → 7 topics)"),
        ("#5a3010", "#f5ede0", "#d8c4a8", "Predictions + Confidence",     ""),
    ]
    arch_col, _ = st.columns([2, 3])
    with arch_col:
        arch_html = "<div>"
        for i, (color, bg, border_c, lbl, sub) in enumerate(arch_items):
            arrow = f'<div style="text-align:center;color:#c0b0a0;font-size:0.85rem;margin:2px 0">↓</div>' if i < len(arch_items) - 1 else ""
            arch_html += (
                f'<div style="background:{bg};border:1px solid {border_c};padding:9px 14px;border-radius:2px;margin-bottom:2px">'
                f'<span style="font-family:Source Sans 3,sans-serif;font-size:0.78rem;font-weight:600;color:{color}">{lbl}</span>'
                + (f'<br><span style="font-family:Source Sans 3,sans-serif;font-size:0.68rem;color:#9a8a74">{sub}</span>' if sub else "")
                + f'</div>{arrow}'
            )
        arch_html += "</div>"
        st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="background:#fffdf9;border:1px solid #d8cebb;border-left:3px solid #c0b0a0;'
        f'padding:12px 18px;border-radius:0 2px 2px 0">'
        f'<span style="font-family:Source Sans 3,sans-serif;font-size:0.75rem;color:#9a8a74">'
        f'Start the backend: <b style="color:#5a3010;font-family:monospace">uvicorn main:app --host 0.0.0.0 --port 8000</b>'
        f'<br>Swagger docs: <b style="color:#5a3010;font-family:monospace">{api_url}/docs</b>'
        f'</span></div>',
        unsafe_allow_html=True,
    )
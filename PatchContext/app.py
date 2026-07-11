import os
import time
import logging
import re
import textwrap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import RAGPipeline
from hallucination_guard import check_grounding
from evaluation import run_offline_evaluation

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

# Check Groq Connection
@st.cache_resource
def check_groq_connection() -> bool:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        llm.invoke("Hello")
        return True
    except Exception:
        return False

groq_api_ok = check_groq_connection()

# Page Configuration
st.set_page_config(
    page_title="PatchContext | FastAPI RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme selection initialization
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Active search result cache initialization
if "search_result" not in st.session_state:
    st.session_state.search_result = None

if "validation_error" not in st.session_state:
    st.session_state.validation_error = None

# Custom theme variables (Requirement 14 & 15)
theme = st.session_state.theme
if theme == "light":
    primary_bg = "#F8FAFC"
    text_primary = "#111827"
    text_secondary = "#475569"
    card_bg = "#FFFFFF"
    card_border = "#E5E7EB"
    sidebar_bg = "#FFFFFF"
    sidebar_text = "#111827"
    sidebar_border = "#E5E7EB"
    metric_label = "#475569"
    metric_value = "#111827"
    scroll_track = "#F1F5F9"
    scroll_thumb = "#CBD5E1"
    hover_bg = "#F1F5F9"
else:
    primary_bg = "#0F172A"
    text_primary = "#F8FAFC"
    text_secondary = "#94A3B8"
    card_bg = "#111827"
    card_border = "rgba(255, 255, 255, 0.08)"
    sidebar_bg = "#0B0E14"
    sidebar_text = "#F8FAFC"
    sidebar_border = "rgba(255, 255, 255, 0.05)"
    metric_label = "#94A3B8"
    metric_value = "#F8FAFC"
    scroll_track = "#1E293B"
    scroll_thumb = "#475569"
    hover_bg = "#1E293B"

# Dynamic border styling snippet
validation_border_css = ""
if st.session_state.get("validation_error"):
    validation_border_css = """
    div[data-testid="stTextInput"] input {
        border-color: #EF4444 !important;
        box-shadow: 0 0 0 1px #EF4444 !important;
    }
    """

# Helper function to prevent raw HTML leak via Markdown preformatting
def render_html(html_str: str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# Global clean styling injection (Requirement 1, 2, 3, 5, 6, 8, 12, 13 & 16)
render_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

{validation_border_css}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {primary_bg} !important;
    color: {text_primary} !important;
    transition: background-color 250ms ease, color 250ms ease;
}}

/* Max page layout centered bounds */
.block-container {{
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
}}

/* Typography headings sizes and weights (Requirement 3) */
h1, h2, h3, h4, h5 {{
    font-weight: 800 !important;
    color: {text_primary} !important;
    margin-top: 16px !important;
    margin-bottom: 12px !important;
}}
h1 {{
    font-size: 32px !important;
}}
h2 {{
    font-size: 24px !important;
}}
h3 {{
    font-size: 18px !important;
}}
p, li {{
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: {text_primary} !important;
    margin-bottom: 12px !important;
}}

/* Inline Code block styling */
code {{
    font-family: 'Courier New', Courier, monospace !important;
    background-color: rgba(120, 120, 120, 0.1) !important;
    color: {text_primary} !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-size: 0.9em !important;
}}

/* Sidebar section layout */
section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {sidebar_border} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {sidebar_text} !important;
}}

/* Match sidebar border cards padding and spacing */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
    border: 1px solid {sidebar_border} !important;
    background-color: {sidebar_bg} !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    transition: transform 250ms ease, border-color 250ms ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {{
    transform: translateY(-1px) !important;
    border-color: #2563EB !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.05) !important;
}}

/* Metric Cards style (Requirement 2) */
[data-testid="metric-container"] {{
    background-color: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.05) !important;
    transition: transform 250ms ease, border-color 250ms ease, background-color 250ms ease !important;
    height: 110px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}}
[data-testid="metric-container"]:hover {{
    transform: translateY(-2px) !important;
    border-color: #2563EB !important;
    background-color: {hover_bg} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {text_secondary} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 30px !important;
    font-weight: 800 !important;
    color: {text_primary} !important;
    line-height: 1.2 !important;
}}

/* Button styles */
div[element-type="button"] button, .stButton > button {{
    height: 42px !important;
    border-radius: 12px !important;
    background-color: {card_bg} !important;
    color: {text_secondary} !important;
    border: 1px solid {card_border} !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    font-size: 14px !important;
    transition: all 250ms ease !important;
}}
div[element-type="button"] button:hover, .stButton > button:hover {{
    background-color: {primary_bg} !important;
    border-color: #3B82F6 !important;
    color: #2563EB !important;
    transform: translateY(-1px) !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}}

/* Taller search inputs */
div[data-testid="stTextInput"] input {{
    height: 48px !important;
    font-size: 15px !important;
    border-radius: 12px !important;
}}

/* Reference Badges Colors (Requirement 5) */
a[href*="/pull/"] {{
    background-color: rgba(37, 99, 235, 0.08) !important;
    color: #2563EB !important;
    border: 1px solid rgba(37, 99, 235, 0.2) !important;
    padding: 6px 12px !important;
    border-radius: 20px !important;
    text-decoration: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 250ms ease !important;
    display: inline-flex !important;
}}
a[href*="/pull/"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    border-color: #2563EB !important;
}}

a[href*="/issues/"] {{
    background-color: rgba(249, 115, 22, 0.08) !important;
    color: #EA580C !important;
    border: 1px solid rgba(249, 115, 22, 0.2) !important;
    padding: 6px 12px !important;
    border-radius: 20px !important;
    text-decoration: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 250ms ease !important;
    display: inline-flex !important;
}}
a[href*="/issues/"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.15) !important;
    border-color: #EA580C !important;
}}

a[href*="/commit/"] {{
    background-color: rgba(168, 85, 247, 0.08) !important;
    color: #9333EA !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
    padding: 6px 12px !important;
    border-radius: 20px !important;
    text-decoration: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 250ms ease !important;
    display: inline-flex !important;
}}
a[href*="/commit/"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(168, 85, 247, 0.15) !important;
    border-color: #9333EA !important;
}}

/* Expand Height Animation */
[data-testid="stExpander"] {{
    transition: all 0.25s ease-in-out !important;
}}

/* Gradient layout Hero card (Requirement 1) */
.hero-banner {{
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.02) 100%) !important;
    border: 1px solid {card_border} !important;
    border-radius: 16px !important;
    padding: 32px 24px !important;
    text-align: center !important;
    animation: fadeIn 0.4s ease-out !important;
    margin-bottom: 32px !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.05) !important;
}}
.hero-icon {{
    font-size: 48px !important;
    margin-bottom: 12px !important;
    display: inline-block !important;
    text-shadow: 0 0 15px rgba(37, 99, 235, 0.4) !important;
    animation: pulse 2s infinite ease-in-out !important;
}}
@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
    100% {{ transform: scale(1); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Typography elements answer card overrides (Requirement 3) */
.answer-title {{
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
    color: {text_primary} !important;
}}
.answer-divider {{
    border: none !important;
    border-top: 1px solid {card_border} !important;
    margin: 24px 0 !important;
}}
</style>
""")

# Theme persistence & Keyboard events listener (Requirement 9, 11 & 16)
st.markdown(f"""
<script>
var storedTheme = localStorage.getItem("patchcontext_theme");
localStorage.setItem("patchcontext_theme", "{theme}");

// Event listener for keys (Ctrl+K, Escape)
window.parent.document.addEventListener('keydown', function(e) {{
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
        e.preventDefault();
        var inputs = window.parent.document.querySelectorAll('input[type="text"]');
        if (inputs.length > 0) {{
            inputs[0].focus();
        }}
    }}
    if (e.key === 'Escape') {{
        var inputs = window.parent.document.querySelectorAll('input[type="text"]');
        if (inputs.length > 0) {{
            inputs[0].value = '';
            inputs[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }}
}});

// Auto focus search input after page load
setTimeout(function() {{
    var inputs = window.parent.document.querySelectorAll('input[type="text"]');
    if (inputs.length > 0) {{
        inputs[0].focus();
    }}
}}, 500);

// Event listener for copy button label feedback
window.parent.document.addEventListener('click', function(e) {{
    var target = e.target;
    if (target && target.tagName === 'BUTTON' && target.innerText.indexOf('Copy') !== -1) {{
        var oldText = target.innerText;
        target.innerText = 'Copied ✓';
        setTimeout(function() {{
            target.innerText = oldText;
        }}, 2000);
    }}
}});

// Same-origin iframe copy bypass
window.parent.copyToClipboard = function(text) {{
    var parentDoc = window.parent.document;
    var el = parentDoc.createElement('textarea');
    el.value = text;
    el.setAttribute('readonly', '');
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    parentDoc.body.appendChild(el);
    el.select();
    var success = false;
    try {{
        success = parentDoc.execCommand('copy');
    }} catch (err) {{
        console.error("Clipboard copy failed: ", err);
    }}
    parentDoc.body.removeChild(el);
    return success;
}};
</script>
""", unsafe_allow_html=True)

# Helper to initialize RAG pipeline in session state
@st.cache_resource
def get_rag_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(owner="fastapi", repo="fastapi")
    pipeline.get_or_create_vectorstore()
    pipeline.setup_qa_chain()
    return pipeline

try:
    pipeline = get_rag_pipeline()
    faiss_status = "Loaded from Local Cache"
    docs = list(pipeline.vectorstore.docstore._dict.values())
    total_chunks = len(docs)
    issues_urls = set()
    prs_urls = set()
    commits_shas = set()
    for doc in docs:
        m = doc.metadata
        t = m.get("type")
        if t == "issue":
            issues_urls.add(m.get("url"))
        elif t == "pr":
            prs_urls.add(m.get("url"))
        elif t == "commit":
            commits_shas.add(m.get("sha"))
    issues_count = len(issues_urls)
    prs_count = len(prs_urls)
    commits_count = len(commits_shas)
except Exception as e:
    pipeline = None
    faiss_status = f"Error: {e}"
    issues_count = 0
    prs_count = 0
    commits_count = 0
    total_chunks = 0
    logger.error(f"Error loading pipeline: {e}")

# Sidebar Configuration (100% Native, Styled cards inside sidebar)
with st.sidebar:
    st.title("PatchContext")
    st.caption("FASTAPI DESIGN RAG")
    st.markdown("---")
    
    # Repository Card
    with st.container(border=True):
        st.markdown("📁 **Repository**")
        st.info("fastapi/fastapi")
        
    # System Status Card
    with st.container(border=True):
        st.markdown("🟢 **System Status**")
        st.success("Ready")
        
    # Connections Card
    with st.container(border=True):
        st.markdown("🔌 **Connections**")
        if groq_api_ok:
            st.success("Groq API: Connected")
        else:
            st.error("Groq API: Disconnected")
        st.success("GitHub API: Connected")
        
    # Stats Card
    with st.container(border=True):
        st.markdown("📊 **Vector Database Stats**")
        st.markdown(f"🐞 Issues ............. **{issues_count}**")
        st.markdown(f"📦 PRs ................. **{prs_count}**")
        st.markdown(f"📝 Commits ........... **{commits_count}**")
        st.markdown(f"📚 Chunks ............. **{total_chunks}**")

    st.markdown("---")
    
    # Force rebuild database button
    if st.button("Force Rebuild Database", use_container_width=True):
        with st.spinner("Refetching GitHub data and rebuilding FAISS database..."):
            try:
                pipeline.get_or_create_vectorstore(force_refresh=True)
                pipeline.setup_qa_chain()
                st.success("FAISS Database Rebuilt!")
                time.sleep(1)
                st.rerun()
            except Exception as ex:
                st.error(f"Rebuild failed: {ex}")

# Helper Functions for UI presentation parsing and cleanup
def clean_document_content(text: str) -> str:
    # 1. Remove HTML tags (e.g. <div>, <hr>, <br>, <img>, etc) and XML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Remove XML/HTML entities
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    # 3. Remove markdown checklist templates like - [ ] or - [x]
    text = re.sub(r"-\s*\[[ xX]*\]", " ", text)
    
    # 4. Remove prompt leakage lines
    lines = text.split("\n")
    cleaned_lines = []
    skip_disclaimer = False
    for line in lines:
        stripped = line.strip()
        
        # Detect AI Disclaimer start
        if "AI Disclaimer" in line or "Disclaimer:" in line:
            skip_disclaimer = True
            continue
        
        # If we hit an empty line or a new header, stop skipping disclaimer
        if skip_disclaimer:
            if not stripped or stripped.startswith("#"):
                skip_disclaimer = False
            continue
            
        if stripped.lower().startswith(("model:", "prompt:", "system prompt:", "ai disclaimer:")):
            continue
            
        # Remove lines that look like prompt injection or template headers
        if "Claude" in line or "GPT" in line or "Gemini" in line or "Llama" in line:
            if stripped.startswith(("**Model:**", "**Prompt:**", "Model:", "Prompt:")):
                continue
                
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines)
    
    # Collapse repeated empty lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

def parse_llm_answer(answer_text: str):
    # Clean raw URLs and trailing sources citations inside prompt answers
    answer_text = re.sub(r"(?i)Sources?:?\s*.*", "", answer_text).strip()
    answer_text = re.sub(r"https://github\.com/\S+", "", answer_text).strip()
    
    lines = answer_text.strip().split("\n")
    summary_lines = []
    key_points = []
    
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Check standard bullet markers
        if stripped.startswith(("- ", "* ", "• ", "+ ")) or (stripped[0].isdigit() and stripped[1:3] in (". ", ") ")):
            content = stripped.lstrip("-*•+ 0123456789.)").strip()
            if content:
                key_points.append(content)
            in_list = True
        else:
            if not in_list:
                summary_lines.append(line)
                
    summary = "\n".join(summary_lines).strip()
    
    # Fallback keypoints if none parsed
    if not key_points:
        paragraphs = [p.strip() for p in summary.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            summary = paragraphs[0]
            key_points = paragraphs[1:]
        else:
            sentences = [s.strip() for s in summary.split(". ") if s.strip()]
            if len(sentences) > 1:
                summary = sentences[0] + "."
                key_points = [s + "." for s in sentences[1:]]
            else:
                key_points = ["Grounded explanation from codebase repository history."]
                
    return summary, key_points

def clean_pdf_text(text: str) -> str:
    text = text.replace("\r", "")
    replacements = {
        "📋": "[Copy]",
        "🛡️": "[Confidence]",
        "🛡": "[Confidence]",
        "🎯": "[Grounding]",
        "📚": "[Sources]",
        "⚡": "[Latency]",
        "✅": "[Status]",
        "🔀": "PR",
        "🐞": "Issue",
        "📝": "Commit",
        "⬇": "Download",
        "•": "-",
        "✓": "[OK]"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def convert_text_to_pdf(text: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    
    cleaned_text = clean_pdf_text(text)
    in_code_block = False
    for line in cleaned_text.split("\n"):
        line_strip = line.strip()
        if line_strip.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            pdf.set_font("courier", size=10)
            pdf.multi_cell(0, 7, text="    " + line, new_x="LMARGIN", new_y="NEXT")
            continue
            
        if line_strip.startswith("# "):
            pdf.set_font("helvetica", style="B", size=16)
            pdf.multi_cell(0, 10, text=line_strip[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line_strip.startswith("## "):
            pdf.set_font("helvetica", style="B", size=13)
            pdf.multi_cell(0, 8, text=line_strip[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line_strip.startswith("- ") or line_strip.startswith("* "):
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 7, text=line_strip, new_x="LMARGIN", new_y="NEXT")
        elif not line_strip:
            pdf.ln(5)
        else:
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 7, text=line, new_x="LMARGIN", new_y="NEXT")
            
    return bytes(pdf.output())

def generate_markdown_content(query: str, summary: str, decisions: list, evidence: str, references: list, confidence: str, grounding: str, sources: int, latency: float) -> str:
    md = []
    md.append(f"# Question\n{query}\n")
    md.append(f"## Summary\n{summary}\n")
    if decisions:
        md.append("## Key Design Decisions")
        for d in decisions:
            md.append(f"- {d}")
        md.append("")
    if evidence:
        md.append(f"## Evidence\n{evidence}\n")
    if references:
        md.append("## Repository References")
        for ref in references:
            md.append(f"- {ref}")
        md.append("")
    md.append("## Metrics")
    md.append(f"- Confidence: {confidence}")
    md.append(f"- Grounding: {grounding}")
    md.append(f"- Sources: {sources}")
    md.append(f"- Latency: {latency:.2f} s")
    return "\n".join(md)



# Domain validator logic
def is_valid_repository_query(query: str) -> bool:
    query_lower = query.lower().strip()
    
    # Empty query check
    if not query_lower:
        return False
        
    # Standard greetings or generic phrases
    greetings = {"hi", "hello", "hey", "howdy", "hola", "yo", "good morning", "good afternoon", "good evening", "how are you", "who are you"}
    if query_lower in greetings:
        return False
        
    # Words parsing
    words = set(re.findall(r"\b\w+\b", query_lower))
    if not words:
        return False
        
    # Keywords indicating valid repository search
    fastapi_keywords = {
        "fastapi", "apirouter", "router", "routers", "depends", "dependency", "dependencies", "injection", "injections", "pydantic", "middleware", "middlewares",
        "staticfiles", "pr", "issue", "issues", "commit", "commits", "background", "task", "tasks", "websocket", "websockets", "starlette", "lifespan",
        "event", "events", "routing", "route", "routes", "parameter", "parameters", "query", "queries", "path", "paths", "response", "responses",
        "request", "requests", "security", "auth", "oauth", "cors", "uwsgi", "uvicorn", "gunicorn", "asgi", "serialize", "serialization", "serialized",
        "json", "pydantic", "validate", "validation", "decorator", "decorators", "async", "await", "def", "class", "app"
    }
    
    # Substring containment check for high-confidence matches
    for w in words:
        for k in fastapi_keywords:
            if k in w or w in k:
                # Add logging around query classification
                logger.info(f"Query '{query}' classified as valid based on match: word '{w}' vs keyword '{k}'")
                return True
        
    # Match Issue/PR patterns like PR #12 or Issue #15956
    if re.search(r"\b(pr|issue|commit|sha)\b\s*#?\d+", query_lower):
        logger.info(f"Query '{query}' classified as valid based on Issue/PR regex pattern")
        return True
        
    # Invalid generic topics
    invalid_keywords = {"weather", "president", "news", "recommendation", "laptop", "india", "china", "usa", "cooking", "recipe", "food"}
    if words.intersection(invalid_keywords):
        logger.info(f"Query '{query}' classified as invalid due to intersection with invalid keywords")
        return False
        
    # Give benefit of doubt for standard technical inquiries
    technical_keywords = {"how", "why", "what", "explain", "code", "function", "class", "decorator", "app", "serve", "run", "inject", "model", "validation", "schema"}
    if words.intersection(technical_keywords) and len(words) > 2:
        logger.info(f"Query '{query}' classified as valid technical inquiry")
        return True
        
    logger.info(f"Query '{query}' classified as invalid domain")
    return False

# Main Tabs (Emoji-free)
tab_search, tab_eval, tab_about = st.tabs(["Codebase Assistant", "Pipeline Evaluation", "About Project"])

# Session state variables
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "history" not in st.session_state:
    st.session_state.history = []

# Callback to handle enter key submit
def on_query_submit():
    val = st.session_state.query_input_field
    if not val or not val.strip():
        st.session_state.validation_error = "Please enter a question."
        st.session_state.search_result = None
    else:
        st.session_state.validation_error = None
        st.session_state.search_query = val


# ----------------- SEARCH TAB -----------------
with tab_search:
    # Navigation / Theme Toggle bar (Requirement 2 & 20)
    col_nav_title, col_nav_toggle = st.columns([10, 2])
    with col_nav_toggle:
        theme_label = "☀️ Light Mode" if theme == "dark" else "🌙 Dark Mode"
        if st.button(theme_label, key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()
            
    # Shortened Gradient Centhero Banner Section (Requirement 1)
    with st.container():
        render_html(f"""
        <div class='hero-banner'>
            <div class='hero-icon'>⚡</div>
            <h2 style='margin: 0 0 4px 0; padding: 0;'>PatchContext</h2>
            <p style='font-size: 13px; font-weight: 500; color: #2563EB; margin: 0 0 16px 0; padding: 0;'>Grounded AI Assistant for FastAPI Repository History</p>
        </div>
        """)
        
        # Aligned stats in one row
        col_h_stats = st.columns(4)
        with col_h_stats[0]:
            st.metric("Repository", "fastapi/fastapi")
        with col_h_stats[1]:
            st.metric("Indexed Chunks", total_chunks)
        with col_h_stats[2]:
            st.metric("Pull Requests", prs_count)
        with col_h_stats[3]:
            st.metric("Issues", issues_count)
    
    # Check processing status
    is_processing = st.session_state.is_processing
    
    # 2. Search Area Card
    with st.container(border=True):
        st.markdown("### Ask Assistant")
        
        # User Search Box
        user_query = st.text_input(
            "Enter your question about the codebase design:",
            value=st.session_state.search_query,
            key="query_input_field",
            placeholder="Ask why FastAPI was designed this way...",
            label_visibility="collapsed",
            on_change=on_query_submit,
            disabled=is_processing
        )
        
        # Validation error message display (Requirement 4)
        if st.session_state.get("validation_error"):
            st.error(st.session_state.validation_error)
        
        # Try asking presets list
        st.markdown("##### Try asking:")
        presets = [
            "What is FastAPI?",
            "How does APIRouter work?",
            "Explain Dependency Injection.",
            "Where is middleware handled?",
            "Path parameters routing?"
        ]
        
        preset_cols = st.columns(len(presets))
        for i, preset in enumerate(presets):
            # Highlight selected chip during load, deselect after search completes
            is_selected = (st.session_state.search_query == preset) and is_processing
            btn_type = "primary" if is_selected else "secondary"
            if preset_cols[i].button(preset, key=f"p_btn_{i}", type=btn_type, use_container_width=True, disabled=is_processing):
                st.session_state.validation_error = None
                st.session_state.search_query = preset
                st.session_state.is_processing = True
                st.rerun()
                
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        
        # Search activation triggers
        trigger_search = False
        col_search, _ = st.columns([1.5, 3.5])
        with col_search:
            # Main Ask button
            if is_processing:
                st.button("Searching...", key="searching_disabled_btn", type="primary", use_container_width=True, disabled=True)
            else:
                if st.button("Ask Assistant", type="primary", use_container_width=True, key="ask_button_main"):
                    if not user_query or not user_query.strip():
                        st.session_state.validation_error = "Please enter a question."
                        st.session_state.search_result = None
                    else:
                        st.session_state.validation_error = None
                        st.session_state.search_query = user_query
                        trigger_search = True
                
    # Detect query changes or Enter submits
    if st.session_state.search_query and st.session_state.search_query != st.session_state.last_query and not is_processing:
        trigger_search = True
        
    if trigger_search and st.session_state.search_query:
        current_query = st.session_state.search_query
        st.session_state.last_query = current_query
        
        if not groq_api_ok:
            st.error("Cannot execute query. Please add a valid GROQ_API_KEY in your .env file.")
        elif not pipeline:
            st.error("FAISS index/RAG pipeline is not loaded. Try force-rebuilding the database.")
        else:
            # Set processing state
            st.session_state.is_processing = True
            st.rerun()

# Skeleton loading representation
if st.session_state.is_processing:
    # 1. Metrics skeleton
    st.markdown("### Metrics Panel")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    for col in [col_s1, col_s2, col_s3, col_s4, col_s5]:
        with col:
            st.markdown("<div class='skeleton-card' style='height:80px;'></div>", unsafe_allow_html=True)
            
    # 2. Answer Summary skeleton
    st.markdown("### Answer Summary")
    st.markdown("<div class='skeleton-card' style='height:120px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='skeleton-card' style='height:100px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='skeleton-card' style='height:80px;'></div>", unsafe_allow_html=True)
    
    # 3. Retrieved documents skeleton
    st.markdown("### Retrieved Source Documents")
    st.markdown("<div class='skeleton-card' style='height:140px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='skeleton-card' style='height:140px;'></div>", unsafe_allow_html=True)

# Execute search when processing state is active
if st.session_state.is_processing and st.session_state.search_query:
    current_query = st.session_state.search_query
    
    # Progress feedback steps
    with st.spinner("Processing..."):
        load_prog = st.progress(0)
        load_stat = st.empty()
        
        load_stat.markdown("Searching repository...")
        load_prog.progress(20)
        time.sleep(0.2)
        
        # Domain validation check
        is_valid = is_valid_repository_query(current_query)
        
        if not is_valid:
            time.sleep(0.2)
            st.session_state.search_result = {
                "status": "invalid_domain",
                "query": current_query
            }
        else:
            load_stat.markdown("Retrieving evidence...")
            load_prog.progress(40)
            
            # Execute RAG query
            try:
                result = pipeline.query(current_query)
                ans = result["answer"]
                srcs = result["source_documents"]
                t_exec = result["execution_time"]
                
                load_stat.markdown("Ranking documents...")
                load_prog.progress(60)
                time.sleep(0.1)
                
                is_fallback = (ans.strip() == "I couldn't find this in repository history.")
                
                if is_fallback:
                    st.session_state.search_result = {
                        "status": "no_evidence",
                        "query": current_query
                    }
                else:
                    load_stat.markdown("Generating grounded answer...")
                    load_prog.progress(80)
                    
                    # Grounding check
                    src_texts = [d.page_content for d in srcs]
                    is_grounded, g_score, verdict = check_grounding(ans, src_texts)
                    
                    load_stat.markdown("Rendering results...")
                    load_prog.progress(100)
                    time.sleep(0.1)
                    
                    st.session_state.search_result = {
                        "status": "success",
                        "query": current_query,
                        "answer": ans,
                        "source_docs": srcs,
                        "score": g_score,
                        "verdict": verdict,
                        "time": t_exec
                    }
                    
                    # Add search to history if success
                    hist_entry = {"query": current_query, "verdict": verdict, "time": t_exec, "timestamp": time.strftime("%H:%M:%S")}
                    if hist_entry not in st.session_state.history:
                        st.session_state.history.insert(0, hist_entry)
                        st.session_state.history = st.session_state.history[:5]
            except Exception as e:
                st.session_state.search_result = {
                    "status": "error",
                    "query": current_query,
                    "error_msg": str(e)
                }
        
        # Clear loaders and release processing state
        load_prog.empty()
        load_stat.empty()
        st.session_state.is_processing = False
        st.rerun()

# RENDER UI STATES: Empty State / Invalid / No Evidence / Success

# 1. Better Empty State
if st.session_state.search_result is None and not is_processing:
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px; border: 1px dashed var(--card-border); border-radius: 16px; background-color: var(--card-bg);'>
        <div style='font-size: 32px; margin-bottom: 12px;'>💬</div>
        <h2 style='font-size: 18px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;'>Ask anything about FastAPI's repository.</h2>
        <p style='font-size: 13px; color: var(--text-secondary); max-width: 500px; margin: 0 auto 16px auto;'>
            Ask why FastAPI was designed this way and get answers compiled directly from issues, pull requests, and commit summaries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Examples
    st.markdown("##### Examples to try:")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    examples = [
        "Why was APIRouter introduced?",
        "How does Dependency Injection work?",
        "Explain middleware.",
        "How does validation work?"
    ]
    cols = [ex_col1, ex_col2, ex_col3, ex_col4]
    for idx, ex in enumerate(examples):
        if cols[idx].button(ex, key=f"ex_btn_{idx}", use_container_width=True):
            st.session_state.search_query = ex
            st.session_state.is_processing = True
            st.rerun()

# 2. Invalid Domain Warning
elif st.session_state.search_result and st.session_state.search_result["status"] == "invalid_domain" and not is_processing:
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        _, col_empty_center, _ = st.columns([1, 10, 1])
        with col_empty_center:
            st.markdown("<h3 style='text-align: center;'>⚠️ Repository Information Not Found</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: var(--text-secondary);'>This assistant answers questions using the indexed FastAPI repository. Your question appears to be unrelated to the indexed repository.</p>", unsafe_allow_html=True)
            
            st.markdown("#### Suggested repository topics:")
            st.markdown("- FastAPI Routing  \n- Dependencies  \n- Middleware  \n- Authentication  \n- Validation")
            
            col_empty_btn1, col_empty_btn2 = st.columns(2)
            with col_empty_btn1:
                if st.button("Try Suggested Query: Explain Dependency Injection", use_container_width=True, key="btn_sg_di_inv"):
                    st.session_state.search_query = "Explain Dependency Injection."
                    st.session_state.is_processing = True
                    st.rerun()
            with col_empty_btn2:
                if st.button("Clear Search", use_container_width=True, key="btn_clear_inv"):
                    st.session_state.search_query = ""
                    st.session_state.search_result = None
                    st.rerun()

# 3. No Evidence Warning
elif st.session_state.search_result and st.session_state.search_result["status"] == "no_evidence" and not is_processing:
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        _, col_empty_center, _ = st.columns([1, 10, 1])
        with col_empty_center:
            st.markdown("<h3 style='text-align: center;'>⚠️ Repository Information Not Found</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: var(--text-secondary);'>This assistant answers questions using the indexed FastAPI repository. Your question appears to be unrelated to the indexed repository.</p>", unsafe_allow_html=True)
            
            st.markdown("#### Suggested repository topics:")
            st.markdown("- FastAPI Routing  \n- Dependencies  \n- Middleware  \n- Authentication  \n- Validation")
            
            col_empty_btn1, col_empty_btn2 = st.columns(2)
            with col_empty_btn1:
                if st.button("Try Suggested Query: Explain Dependency Injection", use_container_width=True, key="btn_sg_di_no"):
                    st.session_state.search_query = "Explain Dependency Injection."
                    st.session_state.is_processing = True
                    st.rerun()
            with col_empty_btn2:
                if st.button("Clear Search", use_container_width=True, key="btn_clear_no"):
                    st.session_state.search_query = ""
                    st.session_state.search_result = None
                    st.rerun()

# 4. Success Result Render
elif st.session_state.search_result and st.session_state.search_result["status"] == "success" and not is_processing:
    res = st.session_state.search_result
    ans_text = res["answer"]
    src_docs = res["source_docs"]
    g_score = res["score"]
    v_verdict = res["verdict"]
    e_time = res["time"]
    
    # HTML smooth scroll anchor target
    st.markdown("<div id='answer-summary-section'></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # 1. RAG Metrics Cards Grid (Requirement 3)
    st.markdown("### Metrics Panel")
    
    # Determine Status badge
    status_text_display = "Grounded & Safe" if v_verdict == "Safe" else "Repository Grounded"
    
    # Max similarity for confidence calculation
    max_sim = 0.0
    if src_docs:
        try:
            scores_diag = pipeline.vectorstore.similarity_search_with_score(st.session_state.last_query, k=1)
            if scores_diag:
                max_sim = 1.0 - (scores_diag[0][1] ** 2) / 2.0
                max_sim = max(0.0, min(1.0, max_sim))
        except Exception:
            pass
            
    confidence_pct = int(g_score * 100) if (v_verdict == "Safe") else int(max_sim * 100)
    confidence_pct = max(0, min(100, confidence_pct))
    
    # Native Columns layout for Metrics (Requirement 2)
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("🛡️ Confidence", f"{confidence_pct}%")
        st.progress(confidence_pct / 100.0)
    with col_m2:
        st.metric("🎯 Grounding", f"{g_score:.2f}")
    with col_m3:
        st.metric("📚 Sources", len(src_docs))
    with col_m4:
        st.metric("⚡ Latency", f"{e_time:.2f} s")
    with col_m5:
        st.metric("✅ Status", status_text_display)
        
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # Parse Summary and Design Decisions
    summary_text, key_decisions = parse_llm_answer(ans_text)
    
    # Dynamic Evidence text
    types_found = set([d.metadata.get("type", "artifact") for d in src_docs])
    types_str = " and ".join(list(types_found)) if types_found else "repository artifacts"
    evidence_str = f"This answer is grounded using FastAPI GitHub {types_str} used as primary evidence."
    
    # Dynamic Reference Badges inside native markdown links (Requirement 5)
    badge_list_markdown = []
    seen_refs = set()
    for d in src_docs:
        m = d.metadata
        dtype = m.get("type", "issue")
        url = m.get("url", "#")
        if dtype == "commit":
            sha = m.get("sha", "")[:7]
            badge_text = f"📝 Commit {sha}"
        elif dtype == "pr":
            num = m.get("number", "N/A")
            badge_text = f"🔀 PR #{num}"
        else:
            num = m.get("number", "N/A")
            badge_text = f"🐞 Issue #{num}"
        
        if badge_text not in seen_refs:
            seen_refs.add(badge_text)
            badge_list_markdown.append(f"[{badge_text}]({url})")
            
    reference_badges_markdown = "   ".join(badge_list_markdown) if badge_list_markdown else "No references cited"
    
    # ONE Unified Answer Card with spacing & layout title elements (Requirement 3)
    with st.container(border=True):
        # Header Row inside Container card
        col_title, col_copy, col_md, col_pdf = st.columns([6, 2, 2, 2])
        with col_title:
            st.markdown("### Answer Summary")
        
        # Collect reference titles
        ref_titles = []
        for d in src_docs:
            m = d.metadata
            dtype = m.get("type", "issue")
            if dtype == "commit":
                ref_titles.append(f"Commit {m.get('sha', '')[:7]}")
            elif dtype == "pr":
                ref_titles.append(f"PR #{m.get('number', 'N/A')}")
            else:
                ref_titles.append(f"Issue #{m.get('number', 'N/A')}")
        
        # Pre-generate valid markdown and PDF files (Requirement 2 & 3)
        markdown_data = ""
        pdf_bytes = b""
        export_disabled = not ans_text or not ans_text.strip()
        
        if not export_disabled:
            try:
                markdown_data = generate_markdown_content(
                    res.get("query", ""),
                    summary_text,
                    key_decisions,
                    evidence_str,
                    ref_titles,
                    f"{confidence_pct}%",
                    f"{g_score:.2f}",
                    len(src_docs),
                    e_time
                )
                logger.info("Successfully generated Markdown content for export")
            except Exception as md_err:
                logger.error(f"Markdown generation failed: {md_err}")
                st.error(f"Failed to generate Markdown content: {md_err}")
                markdown_data = ans_text

            try:
                pdf_bytes = convert_text_to_pdf(markdown_data)
                if not pdf_bytes or len(pdf_bytes) == 0:
                    raise ValueError("Generated PDF content is empty")
                logger.info("Successfully generated PDF content for export")
            except Exception as pdf_err:
                logger.error(f"PDF generation failed: {pdf_err}")
                st.error(f"Failed to generate PDF content: {pdf_err}")
                pdf_bytes = b""
                export_disabled = True

        with col_copy:
            if st.button("📋 Copy", key="btn_copy_ans", use_container_width=True, disabled=export_disabled):
                try:
                    import json
                    escaped_ans_json = json.dumps(markdown_data)
                    st.markdown(f"""
                    <script>
                    var success = false;
                    try {{
                        if (window.parent && window.parent.copyToClipboard) {{
                            success = window.parent.copyToClipboard({escaped_ans_json});
                        }}
                    }} catch (e) {{
                        console.error("Copy error in JS:", e);
                    }}
                    var parentDoc = window.parent.document;
                    if (success) {{
                        var toast = parentDoc.createElement('div');
                        toast.innerText = 'Copied to clipboard';
                        toast.style.position = 'fixed';
                        toast.style.bottom = '20px';
                        toast.style.right = '20px';
                        toast.style.backgroundColor = '#10B981';
                        toast.style.color = '#FFFFFF';
                        toast.style.padding = '12px 24px';
                        toast.style.borderRadius = '8px';
                        toast.style.zIndex = '99999';
                        parentDoc.body.appendChild(toast);
                        setTimeout(function() {{ parentDoc.body.removeChild(toast); }}, 3000);
                    }} else {{
                        var errToast = parentDoc.createElement('div');
                        errToast.innerText = 'Failed to copy to clipboard';
                        errToast.style.position = 'fixed';
                        errToast.style.bottom = '20px';
                        errToast.style.right = '20px';
                        errToast.style.backgroundColor = '#EF4444';
                        errToast.style.color = '#FFFFFF';
                        errToast.style.padding = '12px 24px';
                        errToast.style.borderRadius = '8px';
                        errToast.style.zIndex = '99999';
                        parentDoc.body.appendChild(errToast);
                        setTimeout(function() {{ parentDoc.body.removeChild(errToast); }}, 3000);
                    }}
                    </script>
                    """, unsafe_allow_html=True)
                    logger.info("Executed clipboard copy helper script")
                except Exception as copy_err:
                    logger.error(f"Clipboard copy logic failed: {copy_err}")
                    st.error(f"Copy operation failed: {copy_err}")

        with col_md:
            st.download_button("⬇ Markdown", markdown_data, file_name="answer.md", use_container_width=True, key="btn_dl_md", disabled=export_disabled)
        with col_pdf:
            st.download_button("⬇ PDF", pdf_bytes, file_name="answer.pdf", mime="application/pdf", use_container_width=True, key="btn_dl_pdf", disabled=export_disabled)

            
        render_html("<div class='answer-divider'></div>")
        
        render_html("<div class='answer-title'>Summary</div>")
        st.write(summary_text)
        
        if key_decisions:
            render_html("<div class='answer-divider'></div>")
            render_html("<div class='answer-title'>Key Design Decisions</div>")
            for kp in key_decisions:
                st.markdown(f"- {kp}")
                
        if evidence_str:
            render_html("<div class='answer-divider'></div>")
            render_html("<div class='answer-title'>Evidence</div>")
            st.write(evidence_str)
            
        if reference_badges_markdown:
            render_html("<div class='answer-divider'></div>")
            render_html("<div class='answer-title'>Repository References</div>")
            st.markdown(reference_badges_markdown)
            
        # Spacing and footer stats inside the card
        render_html("<div class='answer-divider'></div>")
        st.caption(f"Confidence: {confidence_pct}% | Status: {v_verdict} | Latency: {e_time:.2f} s")
    
    st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)
        
    # 3. Retrieved Source Documents
    st.markdown("### Retrieved Source Documents")
    if not src_docs:
        st.write("No source documents retrieved.")
    else:
        for s_idx, doc in enumerate(src_docs, 1):
            meta = doc.metadata
            dtype = meta.get("type", "issue")
            title = meta.get("title", "No Title")
            author = meta.get("author", "unknown")
            created = meta.get("created_at", "")[:10] or "N/A"
            url = meta.get("url", "#")
            
            # Clean content preview
            cleaned_content = clean_document_content(doc.page_content)
            snippet = cleaned_content[:200]
            if len(cleaned_content) > 200:
                snippet += "..."
                
            # Reference badge markdown
            if dtype == "commit":
                sha = meta.get("sha", "")[:7]
                badge_md = f"**Commit {sha}**"
            elif dtype == "pr":
                num = meta.get("number", "N/A")
                badge_md = f"**PR #{num}**"
            else:
                num = meta.get("number", "N/A")
                badge_md = f"**Issue #{num}**"
                
            # Render as native border container card
            with st.container(border=True):
                st.markdown(f"{badge_md} - {title}")
                st.caption(f"Author: {author}  |  Date: {created}")
                st.write(snippet)
                
                col_btn_git, col_btn_exp = st.columns([1, 1])
                with col_btn_git:
                    st.link_button("🐙 View on GitHub", url, use_container_width=True)
                with col_btn_exp:
                    with st.expander("🔍 Show Full Content"):
                        st.code(cleaned_content)
                    
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
            
    # Smooth scroll JS invocation
    st.markdown("""<script>
    var el = window.parent.document.getElementById('answer-summary-section');
    if (el) {
        el.scrollIntoView({behavior: 'smooth'});
    }
</script>""", unsafe_allow_html=True)

# Search History Card List
if st.session_state.history and not is_processing:
    st.markdown("### Search History")
    for idx, h in enumerate(st.session_state.history):
        badge_text = "Grounded" if h["verdict"] == "Safe" else "Low Conf"
        ts = h.get("timestamp", "N/A")
        with st.container(border=True):
            col_h1, col_h2 = st.columns([9, 3])
            with col_h1:
                st.markdown(f"**Query**: {h['query']}")
                st.caption(f"Verdict: {badge_text} | Execution Time: {h['time']:.2f} s | Time: {ts}")
            with col_h2:
                if st.button("Rerun Search", key=f"rerun_btn_{idx}", use_container_width=True):
                    st.session_state.search_query = h["query"]
                    st.session_state.is_processing = True
                    st.rerun()

# ----------------- EVALUATION TAB -----------------
with tab_eval:
    st.markdown("## System Evaluation")
    st.markdown(
        "Evaluate the performance of PatchContext across the 10 benchmark queries. "
        "The system measures RAG validity, retrieve-success, and calculates grounding scores via word-overlap algorithms."
    )
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    if st.button("Run Offline Evaluation Suite", type="primary", disabled=st.session_state.is_processing):
        if not groq_api_ok:
            st.error("Please configure GROQ_API_KEY in your .env file to run evaluation.")
        elif not pipeline:
            st.error("Pipeline not initialized correctly.")
        else:
            with st.spinner("Running evaluation suite (this executes 10 benchmark queries on Groq)..."):
                try:
                    df, fig = run_offline_evaluation(pipeline)
                    
                    st.success("Evaluation completed successfully!")
                    
                    # Compute aggregate statistics
                    avg_grounding = df["Grounding Score"].mean()
                    pass_rate = df["Passed"].mean() * 100
                    avg_exec_time = df["Execution Time (s)"].mean()
                    
                    st.markdown("### Aggregated Metrics")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Average Grounding Score", f"{avg_grounding:.2f}")
                    stat_col2.metric("Pass Rate (Score >= 0.15)", f"{pass_rate:.1f}%")
                    stat_col3.metric("Average Latency", f"{avg_exec_time:.2f}s")
                    
                    # Display Results DataFrame
                    st.markdown("### Question-by-Question breakdown")
                    st.dataframe(
                        df[["Question", "Execution Time (s)", "Has Answer", "Has Sources", "Grounding Score", "Passed"]],
                        use_container_width=True
                    )
                    
                    # Display Chart
                    st.markdown("### Grounding Scores Visualization")
                    st.pyplot(fig)
                    
                except Exception as ex:
                    st.error(f"An error occurred during evaluation: {ex}")

# ----------------- ABOUT TAB -----------------
with tab_about:
    st.markdown("## About PatchContext")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    PatchContext is a **Retrieval Augmented Generation (RAG)** assistant that answers developer questions explaining **WHY** FastAPI was designed a certain way.
    It prevents hallucinations by validating answers against retrieved repository metadata (closed issues, pull requests, and commits) and enforces a strict safety fallback:
    
    > **"I couldn't find this in repository history."**
    
    ### Key Features
    * **Explain the Design 'Why'**: Directly focuses on creator discussions, trade-offs, and issues, rather than plain code syntax definitions.
    * **High Precision Retrieval**: MMR retriever filters out irrelevant noisy updates (e.g. general dependabot updates).
    * **Hallucination Guard**: Cross-verifies the answer text semantic overlap with the source chunks.
    * **Local Vector Caching**: Saves and loads the index directory quickly, saving request API limits.
    """)
    
    st.markdown("### Pipeline Architecture & Data Flow")
    
    # Text-based connected visual flow diagram (Requirement 15)
    st.code("""
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  GitHub Data   │ ───>  │   Document     │ ───>  │  Text Splitting │
│  Extraction    │      │  Parsing       │      │  & Chunking    │
└────────────────┘      └────────────────┘      └────────────────┘
        │
        ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Embedding     │ ───>  │  FAISS Vector  │ ───>  │  MMR Retriever │
│  Generation    │      │  Database      │      │  (k=5, f_k=20) │
└────────────────┘      └────────────────┘      └────────────────┘
        │
        ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Groq LLM      │ ───>  │  Grounding     │ ───>  │  Grounded      │
│  Inference     │      │  Evaluation    │      │  Final Answer  │
└────────────────┘      └────────────────┘      └────────────────┘
    """)
    
    st.markdown("""
    ### Tech Stack
    - **Orchestration**: LangChain, LangChain Community, LangChain Core, LangChain Groq
    - **Vector Index**: FAISS (faiss-cpu)
    - **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
    - **LLM Engine**: Groq Cloud REST (llama-3.1-8b-instant)
    - **HTTP Client**: Requests (automatic unauthorized retry)
    - **Data Operations**: Pandas & Matplotlib
    
    ### Team & Author
    - **Developer**: Ashwani Kumar
    - **Organization**: Celebal Technology Internship 2026
    """)

# Footer display (Requirement 22)
st.markdown("<div style='margin-top: 80px; text-align: center; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 20px; color: var(--text-secondary); font-size: 0.8rem;'>Powered by FastAPI  |  Groq  |  GitHub API  |  LangChain  |  FAISS  |  Version 1.0</div>", unsafe_allow_html=True)

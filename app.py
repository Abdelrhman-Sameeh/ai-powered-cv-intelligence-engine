import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pypdf import PdfReader

# 1. Page Configuration
st.set_page_config(
    page_title="CV AI Extractor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-resolution background graphics with dark overlay
BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1920&auto=format&fit=crop"
SIDEBAR_IMAGE_URL = "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=600&auto=format&fit=crop"
BANNER_GRAPHIC_URL = "https://images.unsplash.com/photo-1677442136019-21780efad99a?q=80&w=1200&auto=format&fit=crop"

# 2. Advanced CSS with Background Overlay & Glassmorphism
st.markdown(f"""
<style>
    /* Full Page Background with Overlay Graphic */
    .stApp {{
        background: linear-gradient(
            rgba(11, 15, 25, 0.88), 
            rgba(11, 15, 25, 0.94)
        ), 
        url('{BACKGROUND_IMAGE_URL}') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Sidebar Background Graphic */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(
            rgba(15, 23, 42, 0.92), 
            rgba(15, 23, 42, 0.98)
        ), 
        url('{SIDEBAR_IMAGE_URL}') no-repeat center center;
        background-size: cover;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Glassmorphism Hero Card Header */
    .hero-banner {{
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        padding: 2.5rem 2rem;
        background: linear-gradient(
            135deg, 
            rgba(30, 41, 59, 0.75) 0%, 
            rgba(15, 23, 42, 0.85) 100%
        );
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }}

    .hero-title {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}

    .hero-sub {{
        color: #E2E8F0;
        font-size: 1.1rem;
        max-width: 800px;
        line-height: 1.6;
    }}

    /* Glass Cards for Columns */
    div[data-testid="column"] > div {{
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}

    /* Neon Output Card */
    .output-card {{
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        color: #F8FAFC;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
        line-height: 1.6;
    }}

    /* Glowing Primary CTA Button */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35);
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.55);
        color: #FFFFFF;
    }}

    /* Custom File Uploader & Inputs */
    div[data-testid="stFileUploader"] {{
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
        padding: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return tokenizer, model

# 3. Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain--v1.png", width=64)
    st.title("Control Panel")
    
    with st.status("Loading LLM Engine...", expanded=True) as status:
        tokenizer, model = load_llm()
        status.update(label="AI Engine Active", state="complete")
    
    st.divider()
    st.markdown("### ⚡ System Metrics")
    st.caption(f"**Model:** `{MODEL_NAME}`")
    st.caption("**Precision:** `Float16`")
    st.caption("**Hardware:** `CUDA / GPU`")
    st.caption("**Inference:** `Local Disk Cache`")

# 4. Hero Banner Header
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">CV & Resume Intelligence Engine</div>
    <div class="hero-sub">
        Transform unstructured candidate PDFs into structured key profiles with automated local LLM intelligence.
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Two-Column Dashboard Grid
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 1. Document Input")
    uploaded_file = st.file_uploader("Upload candidate CV (PDF format)", type=["pdf"])

    extracted_text = ""
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        st.success(f"Loaded: **{uploaded_file.name}**")
        
        with st.expander("🔍 Inspect Extracted Raw Text", expanded=False):
            st.text_area("Extracted PDF Content", extracted_text, height=220, label_visibility="collapsed")

with col2:
    st.markdown("### 🧠 2. AI Extraction")
    
    if uploaded_file is None:
        st.info("Upload a candidate CV on the left to begin automated profile extraction.")
    else:
        if st.button("🚀 Extract Structured Resume Profile", type="primary"):
            with st.spinner("Analyzing document structure & parsing entity fields..."):
                prompt = (
                    "Extract key details such as Full Name, Contact Details, Technical Skills, "
                    "Work Experience, and Education from the following CV. Format the output with clean markdown headings and bullet points:\n\n"
                    f"{extracted_text}"
                )
                
                messages = [{"role": "user", "content": prompt}]
                text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=400)
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                st.markdown("#### Extracted Candidate Profile")
                st.markdown(f'<div class="output-card">{response}</div>', unsafe_allow_html=True)

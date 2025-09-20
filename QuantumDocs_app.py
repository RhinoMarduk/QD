# QuantumDocs_app.py
import streamlit as st
st.set_page_config(page_title='QuantumDocs', layout='wide')

import os, io, re, json, glob, zipfile, tempfile
from datetime import datetime
from fractions import Fraction
from typing import List, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import pytesseract

# Optional pdf2image
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# OpenAI optional
try:
    import openai
except Exception:
    openai = None

st.title("🔷 QuantumDocs — Cloud FullSafe")
st.caption("Tesseract OCR · Clause Finder · Mineral/Surface Ledgers · GPT optional")

# Sidebar logo placeholder
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png")

# API key
OPENAI_KEY = ""
if openai:
    try:
        OPENAI_KEY = st.secrets.get("openai_api_key", "")
    except Exception:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    if "openai_api_key_input" not in st.session_state:
        st.session_state.openai_api_key_input = OPENAI_KEY or ""
    val = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input")
    if val.strip():
        OPENAI_KEY = val.strip()
    if OPENAI_KEY:
        openai.api_key = "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"

# OCR helpers
def ocr_tesseract(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return str(e)

def extract_full_text(file_bytes: bytes, mime_type: str) -> str:
    if mime_type == "application/pdf":
        if not convert_from_bytes:
            st.error("PDF support requires pdf2image+Poppler")
            return ""
        try:
            images = convert_from_bytes(file_bytes)
        except Exception as e:
            st.error(str(e)); return ""
    else:
        images = [Image.open(io.BytesIO(file_bytes))]
    return "\n".join([ocr_tesseract(im.convert("RGB")) for im in images])

# Clause detection
CLAUSE_PATTERNS = [
    r"surface\s+only", r"surface\s+estate\s+only",
    r"convey(?:ance)?\s+of\s+(?:the\s+)?surface\s+estate",
    r"reserving", r"hereby reserve", r"less and except", r"save and except",
    r"reserves", r"reserving unto", r"subject to", r"burdened by", r"deducted from"
]

def is_surface_only(text: str) -> bool:
    return any(re.search(pat, text, flags=re.I) for pat in CLAUSE_PATTERNS[:3])

def find_clauses(text: str, extra: str=""):
    pats = CLAUSE_PATTERNS.copy()
    if extra: pats.append(re.escape(extra))
    hits = {}
    for pat in pats:
        found = re.findall(pat, text, flags=re.I)
        if found:
            hits[pat] = found
    return hits

# GPT metadata
def parse_metadata_with_gpt(text: str):
    if not (openai and OPENAI_KEY): return {}
    prompt = "Extract JSON with keys: date_of_document, grantors, grantees, legal_description."
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"{prompt}\n\n{text[:3000]}"}],
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}

# UI
tab1, tab2 = st.tabs(["Single Doc","Batch"])

with tab1:
    upl = st.file_uploader("Upload file", type=["pdf","png","jpg","jpeg","tiff"])
    clause = st.text_input("Extra clause")
    if upl and st.button("Process"):
        txt = extract_full_text(upl.read(), upl.type)
        st.text_area("OCR", txt, height=200)
        st.json(find_clauses(txt, clause))
        meta = parse_metadata_with_gpt(txt)
        st.json(meta)

with tab2:
    zf = st.file_uploader("Upload zip", type=["zip"])
    if zf and st.button("Run Batch"):
        with tempfile.TemporaryDirectory() as td:
            zp = os.path.join(td,"f.zip")
            with open(zp,"wb") as f: f.write(zf.read())
            with zipfile.ZipFile(zp,"r") as z: z.extractall(td)
            files = glob.glob(td+"/**", recursive=True)
            rows=[]
            for p in files:
                if os.path.isfile(p) and p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".pdf")):
                    with open(p,"rb") as fh: data=fh.read()
                    mime="application/pdf" if p.lower().endswith(".pdf") else "image"
                    txt = extract_full_text(data,mime)
                    rows.append({"file":os.path.basename(p),"surface_only":is_surface_only(txt)})
            st.dataframe(pd.DataFrame(rows))

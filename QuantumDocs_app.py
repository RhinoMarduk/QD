
import streamlit as st
st.set_page_config(page_title='QuantumDocs', layout='wide')

from PIL import Image
import pytesseract
import os, io, re, json, glob, zipfile, tempfile
import numpy as np
from fractions import Fraction
from datetime import datetime
import pandas as pd

# Optional libs (no EasyOCR to avoid cv2/torch on Cloud)
try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import boto3
except ImportError:
    boto3 = None
try:
    from google.cloud import vision
except ImportError:
    vision = None
try:
    import networkx as nx
except ImportError:
    nx = None

import openai

# ---- Branding ----
LOGO_PATH = os.path.join('assets', 'logo.png')
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_column_width=True)
st.sidebar.markdown("### QuantumDocs (Cloud‑Safe)")
st.sidebar.caption("Tesseract OCR • Legal Parsing • Ownership Ledgers")

# === Unified OpenAI key handling to avoid duplicate element IDs ===
def _get_openai_key_once():
    if 'openai_api_key_value' not in st.session_state:
        st.session_state.openai_api_key_value = os.getenv("OPENAI_API_KEY", "")
    if 'api_key_ui_mounted' not in st.session_state:
        val = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_input")
        if val:
            st.session_state.openai_api_key_value = val
        st.session_state.api_key_ui_mounted = True
    return st.session_state.openai_api_key_value

openai.api_key = "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"
if not openai.api_key:
    st.sidebar.warning("Add your OpenAI key to enable GPT features.")

# --- OCR (Tesseract only here) ---
def preprocess_pil(pil: Image.Image) -> Image.Image:
    if cv2 is None:
        return pil
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_LINEAR)
    img = cv2.medianBlur(img, 3)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return Image.fromarray(th)

def ocr_tesseract(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def ocr_aws_textract(image_bytes: bytes) -> str:
    if not boto3: return ''
    try:
        client = boto3.client('textract')
        resp = client.detect_document_text(Document={'Bytes': image_bytes})
        return ' '.join([b['Text'] for b in resp.get('Blocks', []) if b['BlockType']=='LINE'])
    except Exception:
        return ''

def ocr_google_vision(image_bytes: bytes) -> str:
    if not vision: return ''
    try:
        client = vision.ImageAnnotatorClient()
        img = vision.Image(content=image_bytes)
        resp = client.document_text_detection(image=img)
        return resp.full_text_annotation.text
    except Exception:
        return ''

def quantum_ocr(image: Image.Image) -> str:
    return ''  # placeholder

def ocr_all(image: Image.Image) -> str:
    img_rgb = image.convert('RGB')
    buf = io.BytesIO(); img_rgb.save(buf, format='JPEG')
    data = buf.getvalue()
    texts = [
        ocr_tesseract(img_rgb),
        ocr_aws_textract(data),
        ocr_google_vision(data),
        quantum_ocr(img_rgb),
    ]
    seen = []
    for t in texts:
        if t and t not in seen:
            seen.append(t)
    return '\\n'.join(seen)

# ---------- Single-file extraction ----------
def extract_full_text(file_bytes: bytes, mime_type: str) -> str:
    images = []
    if mime_type == 'application/pdf':
        if not convert_from_bytes:
            st.error("pdf2image not installed. Run `pip install pdf2image` and install Poppler.")
            return ''
        images = convert_from_bytes(file_bytes)
    else:
        images = [Image.open(io.BytesIO(file_bytes))]
    out = []
    for img in images:
        out.append(ocr_all(preprocess_pil(img)))
    return '\\n'.join(out)

# ---------- GPT helpers ----------
def extract_custom_with_gpt(text: str, fields: list) -> dict:
    if not openai.api_key or not fields:
        return {f: "" for f in fields}
    instruction = (
        "Extract the following custom fields from the document text.\\n"
        "Return STRICT JSON with exactly these keys; use empty string if not present.\\n"
        f"Fields: {fields}\\n\\n"
        f"Document text (truncated):\\n{text[:3500]}"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":instruction}],
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        for f in fields:
            data.setdefault(f, "")
        return {k: ("" if v is None else (str(v) if not isinstance(v, str) else v)) for k, v in data.items()}
    except Exception:
        return {f: "" for f in fields}

def parse_metadata_with_gpt(text: str) -> dict:
    if not openai.api_key:
        return {}
    prompt = (
        "Extract these fields from the legal document text and return strict JSON with keys:"
        " date_of_document, grantors, grantees, legal_description."
        " grantors/grantees as arrays of names; empty string/array if not found.\\n\\n"
        f"Document Text (truncated):\\n{text[:3000]}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    try:
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# ---- Clause finder & Surface-only detection ----
CLAUSE_PATTERNS = [
    r'reserving', r'hereby reserve', r'less and except', r'save and except',
    r'reserves', r'reserving unto', r'subject to', r'burdened by', r'deducted from',
    r'excepting all oil, gas and other minerals',
    r'excepting oil, gas, and other hydrocarbons',
    r'an undivided [\\d/\\.]+\\s*(?:%|percent)?\\s+of (?:his|her|its) right, title, and interest',
    r'surface\\s+only', r'surface\\s+estate\\s+only', r'conveyance\\s+of\\s+surface\\s+estate'
]

def find_clauses(text: str, extra_clause: str) -> dict:
    matches = {}
    pats = CLAUSE_PATTERNS.copy()
    if extra_clause:
        pats.append(re.escape(extra_clause))
    for pat in pats:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            matches[pat] = found
    return matches

def is_surface_only(text: str) -> bool:
    return bool(
        re.search(r'\\b(surface\\s+only|surface\\s+estate\\s+only|convey(?:ance)?\\s+of\\s+the?\\s*surface\\s+estate)\\b', text, re.I) or
        re.search(r'\\bexcluding\\s+all\\s+oil,?\\s*gas.*\\b', text, re.I) or
        re.search(r'\\breserving\\s+all\\s+oil,?\\s*gas.*\\b', text, re.I)
    )

# ---- Ownership Ledgers (same as earlier Cloud-Lite, omitted here for brevity) ----
# (In this Cloud-safe build we keep the core features but avoid EasyOCR.)

st.title('🔷 QuantumDocs — Cloud‑Safe (Tesseract only)')

tab_single, tab_batch = st.tabs(["Single Document", "Batch Mode"])

with tab_single:
    uploaded = st.file_uploader('Upload (PDF/JPG/PNG/TIFF)', type=['pdf','png','jpg','jpeg','tiff'])
    user_clause = st.text_input('Additional clause or phrase')
    custom_spec_single = st.text_area('Custom fields to extract (comma-separated OR JSON dict of field->regex)', height=100, placeholder='lessee, net_acreage\\n-- OR --\\n{"lessee": "Lessee:\\\\s*(.*)", "net_acreage": "Net Acreage:\\\\s*(\\\\d+)"}')
    gpt_query = st.text_input('Ask GPT about the document')

    if st.button('Process Document') and uploaded:
        bytes_data = uploaded.read()
        full_text = extract_full_text(bytes_data, uploaded.type)
        st.subheader('Extracted Text')
        st.text_area('', full_text, height=240)

        # Minimal metadata + clause output to keep build small & safe
        metadata = parse_metadata_with_gpt(full_text)
        st.subheader('Parsed Metadata')
        st.json(metadata)

        clauses = find_clauses(full_text, user_clause)
        st.subheader('Clause Matches')
        for pat, lst in clauses.items():
            st.write(f"{pat}: {lst}")

with tab_batch:
    st.markdown("### Drop a ZIP or multiple image/PDF files")
    zip_file = st.file_uploader('Upload a .zip of pages (recommended)', type=['zip'])
    multi_files = st.file_uploader('Or select multiple files', type=['pdf','png','jpg','jpeg','tiff'], accept_multiple_files=True)
    user_clause_b = st.text_input('Additional clause/phrase to search (batch)')

    def load_images_from_inputs(zip_file, files):
        images = []
        if zip_file is not None:
            with tempfile.TemporaryDirectory() as td:
                zpath = os.path.join(td, 'in.zip')
                with open(zpath, 'wb') as f: f.write(zip_file.read())
                with zipfile.ZipFile(zpath, 'r') as z: z.extractall(td)
                for p in sorted(glob.glob(td+"/**", recursive=True)):
                    if os.path.isfile(p) and p.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                        images.append(Image.open(p).convert('RGB'))
                    elif os.path.isfile(p) and p.lower().endswith('.pdf') and convert_from_bytes:
                        with open(p,'rb') as pf:
                            images.extend(convert_from_bytes(pf.read()))
        for f in (files or []):
            b = f.read()
            if f.type=='application/pdf' and convert_from_bytes:
                images.extend(convert_from_bytes(b))
            else:
                images.append(Image.open(io.BytesIO(b)).convert('RGB'))
        return [preprocess_pil(im) for im in images]

    if st.button('Run Batch'):
        imgs = load_images_from_inputs(zip_file, multi_files)
        if not imgs:
            st.warning('No images found. Upload a ZIP or files.')
        else:
            st.success(f"Loaded {len(imgs)} pages.")
            rows = []
            for im in imgs:
                t = ocr_all(im)
                hits = find_clauses(t, user_clause_b)
                rows.append({
                    'surface_only': bool(is_surface_only(t)),
                    'clauses_found': '; '.join([c for sub in hits.values() for c in sub]),
                    'chars': len(t)
                })
            df = pd.DataFrame(rows)
            st.dataframe(df)


import streamlit as st
st.set_page_config(page_title='QuantumDocs', layout='wide')

from PIL import Image
import pytesseract, easyocr
import os, io, re, json, glob, zipfile, tempfile
import numpy as np
from fractions import Fraction
from datetime import datetime
import pandas as pd

# Optional libs (no PaddleOCR here for Cloud-Lite stability)
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

# --- API key (sidebar) ---
openai_api_key = os.getenv("OPENAI_API_KEY", "")
key_input = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if key_input:
    openai_api_key = key_input
if not openai_api_key:
    st.sidebar.warning("OpenAI API key is missing. Enter it above or set OPENAI_API_KEY.")
openai.api_key = openai_api_key

# --- OCR engines (Tesseract + EasyOCR only) ---
reader_easy = easyocr.Reader(['en'])

def preprocess_pil(pil: Image.Image) -> Image.Image:
    if cv2 is None:
        return pil
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_LINEAR)
    img = cv2.medianBlur(img, 3)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return Image.fromarray(th)

def ocr_pytesseract(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def ocr_easyocr(image: Image.Image) -> str:
    result = reader_easy.readtext(np.array(image))
    return ' '.join([line[1] for line in result])

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
    return ''  # placeholder for future quantum OCR API

def ocr_all(image: Image.Image) -> str:
    img_rgb = image.convert('RGB')
    buf = io.BytesIO(); img_rgb.save(buf, format='JPEG')
    data = buf.getvalue()
    texts = [
        ocr_pytesseract(img_rgb),
        ocr_easyocr(img_rgb),
        ocr_aws_textract(data),
        ocr_google_vision(data),
        quantum_ocr(img_rgb),
    ]
    seen = []
    for t in texts:
        if t and t not in seen:
            seen.append(t)
    return '\n'.join(seen)

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
    return '\n'.join(out)

# ---------- Custom extraction helpers ----------
def parse_custom_spec(spec: str):
    fields, regex_map = [], {}
    if not spec or not spec.strip(): return fields, regex_map
    s = spec.strip()
    if s.startswith('{') or s.startswith('['):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                regex_map = {str(k): str(v) for k, v in obj.items()}
                fields = list(regex_map.keys())
            elif isinstance(obj, list):
                fields = [str(x) for x in obj]
            return fields, regex_map
        except Exception:
            pass
    fields = [x.strip() for x in s.split(',') if x.strip()]
    return fields, regex_map

def extract_custom_regex(text: str, regex_map: dict) -> dict:
    out = {}
    for key, pat in regex_map.items():
        try:
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                out[key] = m.group(1) if m.groups() else m.group(0)
            else:
                out[key] = ""
        except re.error:
            out[key] = ""  # bad pattern
    return out

def extract_custom_with_gpt(text: str, fields: list) -> dict:
    if not openai_api_key or not fields:
        return {f: "" for f in fields}
    instruction = (
        "Extract the following custom fields from the document text.\n"
        "Return STRICT JSON with exactly these keys; use empty string if not present.\n"
        f"Fields: {fields}\n\n"
        f"Document text (truncated):\n{text[:3500]}"
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
    if not openai_api_key:
        return {}
    prompt = (
        "Extract these fields from the legal document text and return strict JSON with keys:"
        " date_of_document, grantors, grantees, legal_description."
        " grantors/grantees as arrays of names; empty string/array if not found.\n\n"
        f"Document Text (truncated):\n{text[:3000]}"
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

# ---- Ownership Ledgers ----
TRACT_FALLBACK = "TRACT-DEFAULT"
SCOPE_ALL = ("ALL_MINERALS", "ALL_DEPTHS")

class BaseBook:
    def __init__(self, label='LEDGER'):
        self.shares = {}
        self.audit = []
        self.tract_acres = {}
        self.warnings = []
        self.label = label

    def set_tract_acres(self, tract, acres):
        try:
            self.tract_acres[tract] = Fraction(int(float(acres)*1000), 1000)
        except Exception:
            self.warnings.append(f"Bad acres value for tract {tract}: {acres}")

    def _key(self, owner, tract, scope, itype):
        return (owner, tract, scope, itype)

    def credit(self, owner, tract, scope, itype, frac, note=""):
        k = self._key(owner, tract, scope, itype)
        self.shares[k] = self.shares.get(k, Fraction(0,1)) + frac
        self.audit.append({"ledger":self.label,"action":"credit","owner":owner,"tract":tract,"scope":scope,"itype":itype,"frac":str(frac),"note":note})

    def debit(self, owner, tract, scope, itype, frac, note=""):
        k = self._key(owner, tract, scope, itype)
        self.shares[k] = self.shares.get(k, Fraction(0,1)) - frac
        self.audit.append({"ledger":self.label,"action":"debit","owner":owner,"tract":tract,"scope":scope,"itype":itype,"frac":str(frac),"note":note})

    def available(self, owner, tract, scope, itype):
        return self.shares.get(self._key(owner, tract, scope, itype), Fraction(0,1))

    def to_dataframe(self):
        rows = []
        for (owner, tract, scope, itype), frac in self.shares.items():
            acres_total = self.tract_acres.get(tract)
            acres_owned = float(frac) * float(acres_total) if acres_total is not None else None
            rows.append({
                'owner': owner, 'tract': tract, 'scope': f"{scope}", 'interest_type': itype,
                'fraction': float(frac), 'fraction_str': str(frac),
                'gross_tract_acres': (float(acres_total) if acres_total is not None else None),
                'owned_acres': acres_owned
            })
        return pd.DataFrame(rows).sort_values(['tract','owner','interest_type'])

class MineralBook(BaseBook): pass
class SurfaceBook(BaseBook): pass

def parse_fraction(text: str) -> Fraction:
    if not text: return Fraction(0,1)
    s = text.strip().lower().replace('%',' percent')
    m = re.search(r"(\\d+)\\s*/\\s*(\\d+)", s)
    if m: return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(percent|%)", s)
    if m: return Fraction(int(float(m.group(1))*10000), 10000*100)
    m = re.search(r"(\\d+(?:\\.\\d+)?)", s)
    if m: return Fraction(int(float(m.group(1))*10000), 10000)
    return Fraction(0,1)

# Events
FIRST_PAGE_PAT = re.compile(r"(WARRANTY DEED|MINERAL DEED|QUIT ?CLAIM|ASSIGNMENT|LEASE|PATENT|RELEASE|AFFIDAVIT|EASEMENT)", re.I)
INSTR_PAT      = re.compile(r"(Instrument No\\.?\\s*\\d+|Doc(?:ument)? #\\s*\\d+|Vol(?:ume)?\\s*\\d+\\s*Pg(?:age)?\\s*\\d+)", re.I)
PAGINATION_PAT = re.compile(r"Page\\s+\\d+\\s+of\\s+\\d+", re.I)
INDEX_HDR_PAT  = re.compile(r"(Grantor|Grantee|Instr|Instrument|Vol\\.?|Pg|Recorded|Date)", re.I)

def regex_events_from_text(text: str, tract: str) -> list:
    evts = []
    surface_flag = is_surface_only(text)

    # ROOT / PATENT detection
    if re.search(r"\\bPATENT\\b|\\bGRANT\\s+FROM\\s+STATE\\b", text, re.I):
        m = re.search(r"(?:to|unto)\\s+([A-Z][A-Za-z.,'\\-\\s]+)", text)
        patentee = m.group(1).strip() if m else "UNKNOWN PATENTEE"
        evts.append({"type":"ROOT_PATENT","grantee":patentee,"interest_type":"MIR","fraction":"1/1","tract":tract,"scope":SCOPE_ALL})

    # RESERVE (basic)
    for m in re.finditer(r"reserves?\\s+(an\\s+undivided\\s+)?([\\d/\\.]+\\s*%?)\\s+(?:of\\s+)?(all\\s+oil,\\s*gas.*|minerals|oil\\s+and\\s+gas|their\\s+right,\\s*title,\\s*and\\s*interest|rti)", text, re.I):
        evts.append({"type":"RESERVE","fraction":str(parse_fraction(m.group(2))),"scope":SCOPE_ALL,"interest_type":"MIR"})

    # UND of RTI
    for m in re.finditer(r"an\\s+undivided\\s+([\\d/\\.]+\\s*%?)\\s+of\\s+(?:his|her|its)\\s+(?:right,\\s*title,\\s*and\\s*interest|rti)", text, re.I):
        evts.append({"type":"UNDT_RTI","fraction":str(parse_fraction(m.group(1))),"scope":SCOPE_ALL,"interest_type":"MIR"})

    # UND by ACRES
    m_ac = re.search(r"an?\\s+undivided\\s+([\\d,]+(?:\\.\\d+)?)\\s*acre", text, re.I)
    if m_ac:
        acres = m_ac.group(1).replace(',', '')
        evts.append({"type":"UNDT_ACRES","acre_amount":acres,"interest_type":"MIR","scope":SCOPE_ALL,"tract":tract})

    # CONVEY
    m_names = re.search(r"between\\s+(.*?)\\s+and\\s+(.*?)[,\\n]", text, re.I)
    if m_names:
        grantor = m_names.group(1).strip()
        grantee = m_names.group(2).strip()
        evts.append({"type":"CONVEY","grantor":grantor,"grantee":grantee,"fraction":"1/1","interest_type":"MIR","scope":SCOPE_ALL,"tract":tract,"surface_only":surface_flag})
    return evts

def parse_events_with_gpt(text: str, tract: str) -> list:
    if not openai_api_key:
        return regex_events_from_text(text, tract)
    prompt = (
        "From this legal document, extract a list of transfer/reservation events as JSON array.\n"
        "Each event object keys: type(one of ROOT_PATENT, CONVEY, RESERVE, EXCEPT, UNDT_RTI, UNDT_ACRES), "
        "grantor(optional), grantee(optional), fraction(string like '1/2' or '50%'), "
        "acre_amount(number if 'undivided X acre' language appears), surface_only(boolean if SURFACE ONLY), "
        "interest_type(one of MIR, RI, ORRI, WI), tract(string), scope(tuple description or 'ALL'), date(ISO if found).\n"
        "If unsure, omit that field. Keep scope 'ALL' if not specified.\n\n"
        f"Document text (truncated):\n{text[:4000]}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    try:
        arr = json.loads(resp.choices[0].message.content)
        if isinstance(arr, list):
            for e in arr:
                e.setdefault('tract', tract)
                e.setdefault('scope', SCOPE_ALL)
            return arr
    except Exception:
        pass
    return regex_events_from_text(text, tract)

def apply_event_to_book(book: BaseBook, e: dict):
    et = e.get('type'); itype = e.get('interest_type','MIR')
    tract = e.get('tract', TRACT_FALLBACK)
    scope = tuple(e.get('scope', SCOPE_ALL)) if isinstance(e.get('scope'), (list,tuple)) else SCOPE_ALL
    frac = parse_fraction(e.get('fraction','1/1'))

    if et == 'ROOT_PATENT':
        gr = e.get('grantee','UNKNOWN')
        book.credit(gr, tract, scope, itype, frac, note='ROOT_PATENT')
        return

    if et in ('RESERVE','UNDT_RTI'):
        return  # simplified prototype

    if et == 'CONVEY':
        grantor = e.get('grantor','UNKNOWN'); grantee = e.get('grantee','UNKNOWN')
        cur = book.available(grantor, tract, scope, itype)
        if cur <= Fraction(0,1): return
        give = None
        acre_amount = e.get('acre_amount')
        if acre_amount:
            try:
                acres = Fraction(int(float(str(acre_amount).replace(',', ''))*1000), 1000)
                tract_total = book.tract_acres.get(tract)
                if tract_total:
                    give = (acres / tract_total)
                    if give > cur: give = cur
                else:
                    book.warnings.append(f"Missing gross acres for tract {tract}; cannot convert {acre_amount} acres to fraction.")
            except Exception:
                book.warnings.append(f"Could not parse acre_amount '{acre_amount}' on tract {tract}.")
        if give is None:
            give = cur if frac == Fraction(1,1) else cur * frac
        book.debit(grantor, tract, scope, itype, give, note='CONVEY')
        book.credit(grantee, tract, scope, itype, give, note='CONVEY')
        return

    if et == 'UNDT_ACRES':
        acres = e.get('acre_amount'); grantee = e.get('grantee','UNKNOWN')
        try:
            acres_val = Fraction(int(float(str(acres).replace(',', ''))*1000), 1000)
            tract_total = book.tract_acres.get(tract)
            if tract_total:
                frac_equiv = acres_val / tract_total
                book.credit(grantee, tract, scope, itype, frac_equiv, note='UNDT_ACRES')
            else:
                book.warnings.append(f"UNDT_ACRES but no gross acres for tract {tract}.")
        except Exception:
            book.warnings.append(f"Bad UNDT_ACRES value: {acres}")

# --- Grouping helpers ---
def quick_ocr_for_grouping(pil: Image.Image) -> str: return pytesseract.image_to_string(pil)
def is_index_page(text: str) -> bool:
    lines = text.splitlines()[:25]
    hits = sum(1 for ln in lines if re.search(r'(Grantor|Grantee|Instr|Instrument|Vol\\.?|Pg|Recorded|Date)', ln, re.I))
    return hits >= 2
def is_first_page(text: str) -> bool:
    return bool(re.search(r"(WARRANTY DEED|MINERAL DEED|QUIT ?CLAIM|ASSIGNMENT|LEASE|PATENT|RELEASE|AFFIDAVIT|EASEMENT)", text, re.I) or re.search(r"(Instrument No\\.?\\s*\\d+|Doc(?:ument)? #\\s*\\d+|Vol(?:ume)?\\s*\\d+\\s*Pg(?:age)?\\s*\\d+)", text, re.I))
def is_last_page(text: str) -> bool:
    return bool(re.search(r"Page\\s+\\d+\\s+of\\s+\\d+", text, re.I))

def group_pages_into_docs(images: list) -> list:
    docs, current = [], []
    for pil in images:
        txt = quick_ocr_for_grouping(pil)
        if is_index_page(txt): continue
        if (not current) or is_first_page(txt):
            if current: docs.append(current)
            current = [(pil, txt)]
        else:
            current.append((pil, txt))
            if is_last_page(txt): docs.append(current); current = []
    if current: docs.append(current)
    return docs

def classify_instrument(text: str) -> str:
    m = re.search(r"(WARRANTY DEED|MINERAL DEED|QUIT ?CLAIM|ASSIGNMENT|LEASE|PATENT|RELEASE|AFFIDAVIT|EASEMENT)", text, re.I)
    return (m.group(1).upper().replace(' ', '_') if m else 'OTHER')

def extract_fields_quick(text: str) -> dict:
    date = ''
    m_date = re.search(r"(\\b\\w+\\s+\\d{1,2},\\s+\\d{4}\\b|\\b\\d{1,2}/\\d{1,2}/\\d{2,4}\\b)", text)
    if m_date: date = m_date.group(1)
    return {
        "date_of_document": date,
        "grantors": "",
        "grantees": "",
        "legal_description": "",
        "instrument_line": (re.search(r'(Instrument No\\.?\\s*\\d+|Doc(?:ument)? #\\s*\\d+|Vol(?:ume)?\\s*\\d+\\s*Pg(?:age)?\\s*\\d+)', text, re.I).group(0) if re.search(r'(Instrument No\\.?\\s*\\d+|Doc(?:ument)? #\\s*\\d+|Vol(?:ume)?\\s*\\d+\\s*Pg(?:age)?\\s*\\d+)', text, re.I) else "")
    }

def build_graph(rows: list):
    if nx is None: return None
    G = nx.DiGraph()
    for r in rows:
        grantors = r.get('grantors', []) if isinstance(r.get('grantors'), list) else []
        grantees = r.get('grantees', []) if isinstance(r.get('grantees'), list) else []
        label = (r.get('instrument_type','') + ' ' + r.get('instrument_line','')).strip()
        for a in grantors:
            for b in grantees:
                if a and b:
                    G.add_edge(a, b, label=label)
    return G

# ---- UI ----
st.title('🔷 QuantumDocs (Cloud‑Lite): Multi‑OCR, Batch & Ownership (No PaddleOCR)')

tab_single, tab_batch = st.tabs(["Single Document", "Batch Mode (Folder/ZIP)"])

with tab_single:
    uploaded = st.file_uploader('Upload (PDF/JPG/PNG/TIFF)', type=['pdf','png','jpg','jpeg','tiff'])
    user_clause = st.text_input('Additional clause or phrase')
    custom_spec_single = st.text_area('Custom fields to extract (comma-separated OR JSON dict of field->regex)', height=100, placeholder='lessee, net_acreage\n-- OR --\n{"lessee": "Lessee:\\s*(.*)", "net_acreage": "Net Acreage:\\s*(\\d+)"}')
    gpt_query = st.text_input('Ask GPT about the document')

    if st.button('Process Document') and uploaded:
        bytes_data = uploaded.read()
        full_text = extract_full_text(bytes_data, uploaded.type)
        st.subheader('Extracted Text')
        st.text_area('', full_text, height=240)

        fields, regex_map = parse_custom_spec(custom_spec_single)
        custom_vals = {}
        if regex_map:
            custom_vals.update(extract_custom_regex(full_text, regex_map))
        remaining = [f for f in fields if f not in custom_vals]
        if remaining:
            custom_vals.update(extract_custom_with_gpt(full_text, remaining))

        metadata = parse_metadata_with_gpt(full_text)
        st.subheader('Parsed Metadata')
        st.json(metadata)

        clauses = find_clauses(full_text, user_clause)
        st.subheader('Clause Matches')
        for pat, lst in clauses.items():
            st.write(f"{pat}: {lst}")

        row = {
            'Filename': uploaded.name,
            'date_of_document': metadata.get('date_of_document',''),
            'grantors': ', '.join(metadata.get('grantors',[])) if isinstance(metadata.get('grantors'), list) else metadata.get('grantors',''),
            'grantees': ', '.join(metadata.get('grantees',[])) if isinstance(metadata.get('grantees'), list) else metadata.get('grantees',''),
            'legal_description': metadata.get('legal_description',''),
            'clauses_found': '; '.join([c for sub in clauses.values() for c in sub])
        }
        for k,v in custom_vals.items():
            row[f"custom_{k}"] = v
        df = pd.DataFrame([row])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        st.download_button('Download Excel', buf.getvalue(), 'quantumdocs_extraction.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    if st.button('Ask GPT') and uploaded:
        if 'full_text' not in st.session_state:
            st.session_state.full_text = extract_full_text(uploaded.read(), uploaded.type)
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {'role':'system','content':'You are a legal document assistant.'},
                {'role':'user','content':f"Document:\n{st.session_state.full_text}\nQuestion: {gpt_query}"}
            ],
            temperature=0.2
        )
        st.subheader('GPT Response')
        st.write(resp.choices[0].message.content)

with tab_batch:
    st.markdown("### Drop a ZIP or multiple image/PDF files")

    # Surface-only view toggle
    c1, c2 = st.columns(2)
    with c1: show_only_surface = st.checkbox('Show only Surface-Only documents', value=False)
    with c2: hide_surface = st.checkbox('Hide Surface-Only documents', value=False, help='If both toggles are off, all docs are shown.')

    st.markdown('#### Custom fields to extract (batch)')
    custom_spec_batch = st.text_area('Comma-separated OR JSON dict of field->regex', key='custom_spec_batch', height=90, placeholder='assignor, assignee\n-- OR --\n{"assignee": "Assignee:\\s*(.*)"}')

    st.markdown('#### Seed Ownership (optional)')
    col1, col2, col3, col4 = st.columns([1.2,1,1,1])
    with col1:
        seed_owner = st.text_input('Seed Owner / Patentee', key='seed_owner', placeholder='e.g., John Smith')
    with col2:
        seed_tract = st.text_input('Tract ID', key='seed_tract', placeholder='e.g., TRACT-DEFAULT')
    with col3:
        seed_interest = st.selectbox('Interest', options=['MIR','RI','ORRI','WI'], index=0, key='seed_interest')
    with col4:
        seed_fraction_str = st.text_input('Fraction', value='1/1', key='seed_fraction')
    st.caption('If provided, the ledgers will start with this owner having the given fraction on ALL_MINERALS/ALL_DEPTHS.')

    st.markdown('##### Multiple Seeds (optional)')
    seeds_csv = st.file_uploader('Upload CSV with columns: tract,owner,interest,fraction', type=['csv'], key='seeds_csv')
    seeds_text = st.text_area('Or paste seeds (one per line: tract,owner,interest,fraction)', key='seeds_text', height=80, placeholder='TRACT-1,John Smith,MIR,1/1\nTRACT-2,ACME LLC,RI,1/8')

    st.markdown('#### Tract Acres Registry (override optional)')
    acres_csv = st.file_uploader('CSV: tract,acres', type=['csv'])
    acres_text = st.text_area('Or paste (tract,acres per line)', height=70, placeholder='TRACT-DEFAULT,640')

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
            prog = st.progress(0)
            st.write(f"Loaded {len(imgs)} pages. Grouping into documents…")
            docs = group_pages_into_docs(imgs)
            rows = []
            events_by_doc = []
            N = len(docs)
            tract_acres_map = {}

            for i, pages in enumerate(docs, 1):
                all_text = '\\n'.join([ocr_all(p[0]) for p in pages])
                quick = extract_fields_quick(all_text)

                fields_b, regex_map_b = parse_custom_spec(custom_spec_batch)
                custom_vals_b = {}
                if regex_map_b:
                    custom_vals_b.update(extract_custom_regex(all_text, regex_map_b))
                remaining_b = [f for f in fields_b if f not in custom_vals_b]
                if remaining_b:
                    custom_vals_b.update(extract_custom_with_gpt(all_text, remaining_b))

                meta = parse_metadata_with_gpt(all_text) if openai_api_key else {}
                tract = TRACT_FALLBACK
                merged = {
                    'instrument_type': classify_instrument(all_text),
                    **quick,
                    **({
                        'date_of_document': meta.get('date_of_document', quick['date_of_document']),
                        'grantors': meta.get('grantors', []),
                        'grantees': meta.get('grantees', []),
                        'legal_description': meta.get('legal_description', quick['legal_description'])
                    } if meta else {})
                }

                # Detect patent acres and record
                if re.search(r'\\bPATENT\\b', all_text, re.I):
                    m_acres_pat = re.search(r'(containing|comprising)\\s+([\\d,]+(?:\\.\\d+)?)\\s*acres', all_text, re.I)
                    if m_acres_pat:
                        try:
                            acres_val = float(m_acres_pat.group(2).replace(',', ''))
                            tract_acres_map[tract] = acres_val
                        except Exception:
                            pass

                # Ownership events
                evts = parse_events_with_gpt(all_text, tract)

                # Attach undivided acres to first CONVEY
                m_ac = re.search(r"an?\\s+undivided\\s+([\\d,]+(?:\\.\\d+)?)\\s*acre", all_text, re.I)
                if m_ac:
                    acres_str = m_ac.group(1); attached = False
                    for e in evts:
                        if e.get('type') == 'CONVEY':
                            e['acre_amount'] = acres_str; e['undivided_acres'] = True; attached = True; break
                    if not attached:
                        evts.append({'type':'UNDT_ACRES', 'acre_amount': acres_str, 'interest_type':'MIR', 'tract': tract, 'scope': SCOPE_ALL})

                clause_hits = find_clauses(all_text, user_clause_b)
                merged['clauses_found'] = '; '.join([c for sub in clause_hits.values() for c in sub])
                merged['page_count'] = len(pages)
                merged['surface_only'] = bool(is_surface_only(all_text))
                rows.append(merged)

                try:
                    d = merged.get('date_of_document','')
                    dt = datetime.strptime(d, '%B %d, %Y') if d and ',' in d else datetime.max
                except Exception:
                    dt = datetime.max
                events_by_doc.append({'date': dt, 'events': evts})
                prog.progress(i/max(N,1))

            df = pd.DataFrame(rows)
            # Apply surface-only filter toggle
            if show_only_surface:
                df_view = df[df['surface_only'] == True]
            elif hide_surface:
                df_view = df[df['surface_only'] != True]
            else:
                df_view = df

            st.subheader('Batch Results')
            st.dataframe(df_view)
            bbuf = io.BytesIO()
            with pd.ExcelWriter(bbuf, engine='openpyxl') as w: df_view.to_excel(w, index=False)
            st.download_button('Download Batch Excel (filtered view)', bbuf.getvalue(), 'quantumdocs_batch_filtered.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # --- Ledgers ---
            st.markdown("### Computed Ownership (Prototype)")
            bookM = MineralBook(label='MINERAL')
            bookS = SurfaceBook(label='SURFACE')

            # Seeds (apply to minerals; optional checkbox to seed surface too)
            if seed_owner and seed_fraction_str:
                try: seed_frac = parse_fraction(seed_fraction_str)
                except Exception: seed_frac = Fraction(1,1)
                tract_seed = seed_tract.strip() or TRACT_FALLBACK
                bookM.credit(seed_owner.strip(), tract_seed, SCOPE_ALL, seed_interest, seed_frac, note='SEEDED')
                if st.checkbox('Also seed Surface Estate with same owner/fraction?', value=True):
                    bookS.credit(seed_owner.strip(), tract_seed, SCOPE_ALL, 'SURFACE', seed_frac, note='SEEDED')

            # Multi-seeds for mineral
            multi_seed_rows = []
            if seeds_csv is not None:
                try:
                    _dfcsv = pd.read_csv(seeds_csv)
                    for _, r in _dfcsv.iterrows():
                        multi_seed_rows.append({
                            'tract': str(r.get('tract', '') or '').strip() or TRACT_FALLBACK,
                            'owner': str(r.get('owner', '') or '').strip(),
                            'interest': str(r.get('interest', 'MIR') or 'MIR').strip().upper(),
                            'fraction': str(r.get('fraction', '1/1') or '1/1').strip()
                        })
                except Exception as e:
                    st.warning(f'Could not parse CSV seeds: {e}')
            if seeds_text:
                for line in seeds_text.splitlines():
                    if not line.strip(): continue
                    try:
                        t,o,i,f = [x.strip() for x in line.split(',')]
                        multi_seed_rows.append({'tract': t or TRACT_FALLBACK, 'owner': o, 'interest': (i or 'MIR').upper(), 'fraction': f})
                    except Exception:
                        st.warning(f'Bad seed line (expected 4 comma-separated values): {line}')
            for r in multi_seed_rows:
                if r['owner']:
                    try: frac = parse_fraction(r['fraction'])
                    except Exception: frac = Fraction(1,1)
                    bookM.credit(r['owner'], r['tract'], SCOPE_ALL, r['interest'], frac, note='SEEDED_MULTI')

            # Apply tract acres maps & overrides
            for t, a in tract_acres_map.items():
                bookM.set_tract_acres(t, a); bookS.set_tract_acres(t, a)
            if acres_csv is not None:
                try:
                    _adf = pd.read_csv(acres_csv)
                    for _, r in _adf.iterrows():
                        t = str(r.get('tract','')).strip() or TRACT_FALLBACK
                        a = r.get('acres', None)
                        if a is not None: bookM.set_tract_acres(t, a); bookS.set_tract_acres(t, a)
                except Exception as e:
                    st.warning(f'Could not parse acres CSV: {e}')
            if acres_text:
                for line in acres_text.splitlines():
                    if not line.strip(): continue
                    try:
                        t, a = [x.strip() for x in line.split(',', 1)]
                        bookM.set_tract_acres(t or TRACT_FALLBACK, a); bookS.set_tract_acres(t or TRACT_FALLBACK, a)
                    except Exception:
                        st.warning(f'Bad acres line: {line}')
            if TRACT_FALLBACK not in bookM.tract_acres:
                bookM.set_tract_acres(TRACT_FALLBACK, 640); bookS.set_tract_acres(TRACT_FALLBACK, 640)

            # Apply events chronologically
            for pkg in sorted(events_by_doc, key=lambda x: x['date']):
                for e in pkg['events']:
                    if e.get('type') == 'ROOT_PATENT':
                        apply_event_to_book(bookM, e)
                        e_surface = e.copy(); e_surface['interest_type'] = 'SURFACE'
                        apply_event_to_book(bookS, e_surface)
                    elif e.get('surface_only'):
                        e2 = e.copy(); e2['interest_type'] = 'SURFACE'
                        apply_event_to_book(bookS, e2)
                        # Note ignored on mineral side
                        bookM.audit.append({"ledger":"MINERAL","action":"surface_convey_ignored","owner":e.get('grantee','UNKNOWN'),"tract":e.get('tract',TRACT_FALLBACK),"scope":e.get('scope',SCOPE_ALL),"itype":"MIR","frac":"0","note":"SURFACE ONLY conveyance ignored for mineral ledger"})
                    else:
                        apply_event_to_book(bookM, e)

            # Render ledgers
            left, right = st.columns(2)
            with left:
                st.subheader("Mineral Ledger")
                dfM = bookM.to_dataframe()
                if not dfM.empty:
                    st.dataframe(dfM)
                    obuf = io.BytesIO()
                    with pd.ExcelWriter(obuf, engine='openpyxl') as w: dfM.to_excel(w, index=False)
                    st.download_button('Download Mineral Ledger (Excel)', obuf.getvalue(), 'quantumdocs_mineral_ledger.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                else:
                    st.info('No mineral ownership changes detected.')
            with right:
                st.subheader("Surface Estate Ledger")
                dfS = bookS.to_dataframe()
                if not dfS.empty:
                    st.dataframe(dfS)
                    sbuf = io.BytesIO()
                    with pd.ExcelWriter(sbuf, engine='openpyxl') as w: dfS.to_excel(w, index=False)
                    st.download_button('Download Surface Ledger (Excel)', sbuf.getvalue(), 'quantumdocs_surface_ledger.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                else:
                    st.info('No surface ownership changes detected.')

            st.markdown("#### Audit Trail (Mineral + Surface)")
            audit_all = pd.DataFrame(bookM.audit + bookS.audit)
            if not audit_all.empty:
                st.dataframe(audit_all)
                abuf = io.BytesIO()
                with pd.ExcelWriter(abuf, engine='openpyxl') as w: audit_all.to_excel(w, index=False)
                st.download_button('Download Audit (Excel)', abuf.getvalue(), 'quantumdocs_audit_all.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                st.download_button('Download Audit (JSON)', json.dumps(bookM.audit + bookS.audit, indent=2).encode('utf-8'), 'quantumdocs_audit_all.json', 'application/json')
            else:
                st.info('No audit entries recorded.')

            # Optional graph
            if nx is not None:
                try:
                    G = build_graph(rows)
                    if G and len(G.nodes) > 0:
                        import matplotlib.pyplot as plt
                        pos = nx.spring_layout(G, seed=42)
                        fig = plt.figure(figsize=(10,10))
                        nx.draw(G, pos, with_labels=True, node_size=800, font_size=8, arrows=True)
                        edge_labels = nx.get_edge_attributes(G, 'label')
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
                        st.pyplot(fig)
                except Exception as e:
                    st.info(f"Graph render skipped: {e}")

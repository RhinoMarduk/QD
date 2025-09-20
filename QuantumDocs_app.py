
# QuantumDocs_app.py — Full Cloud-Safe App
import streamlit as st
st.set_page_config(page_title='QuantumDocs', layout='wide')

import os, io, re, json, glob, zipfile, tempfile
from datetime import datetime
from fractions import Fraction
from typing import List, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import pytesseract

# Optional pdf2image for PDFs
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# Optional OpenAI (not required to run app)
try:
    import openai
except Exception:
    openai = None

# ---------- Branding ----------
st.title('🔷 QuantumDocs — Cloud FullSafe')
st.caption('Tesseract OCR · Heuristic metadata & clauses · Mineral/Surface ledgers · Excel exports')

if os.path.exists('assets/logo.png'):
    st.sidebar.image('assets/logo.png', use_column_width=True)

# ---------- OpenAI (optional) ----------
OPENAI_KEY = ''
if openai:
    try:
        OPENAI_KEY = st.secrets.get('openai_api_key', '')
    except Exception:
        OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
    if 'openai_api_key_input' not in st.session_state:
        st.session_state.openai_api_key_input = OPENAI_KEY or ''
    val = st.sidebar.text_input('OpenAI API Key (optional)', type='password', key='openai_api_key_input')
    if val.strip():
        OPENAI_KEY = val.strip()
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY

# ---------- OCR helpers ----------
def ocr_tesseract(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f'Tesseract error: {e}'

def extract_full_text(file_bytes: bytes, mime_type: str) -> str:
    if mime_type == 'application/pdf':
        if not convert_from_bytes:
            st.error('PDF support requires pdf2image + Poppler')
            return ''
        try:
            images = convert_from_bytes(file_bytes)
        except Exception as e:
            st.error(str(e)); return ''
    else:
        images = [Image.open(io.BytesIO(file_bytes))]
    return '\n'.join([ocr_tesseract(im.convert('RGB')) for im in images])

# ---------- Heuristic metadata ----------
DATE_PAT = re.compile(r"(\b\w+\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)")
GRANTOR_PAT = re.compile(r"\bgrantor[s]?:\s*(.+)", re.I)
GRANTEE_PAT = re.compile(r"\bgrantee[s]?:\s*(.+)", re.I)
LEGAL_DESC_PAT = re.compile(r"\blegal\s+description[:\-\s]*([\s\S]{0,800})", re.I)

def parse_metadata_heuristic(text: str) -> dict:
    date = DATE_PAT.search(text)
    grantors = []
    grantees = []
    m_gor = GRANTOR_PAT.search(text)
    m_gee = GRANTEE_PAT.search(text)
    if m_gor:
        grantors = [x.strip(" .;,:") for x in re.split(r";|,|\band\b", m_gor.group(1)) if x.strip()]
    if m_gee:
        grantees = [x.strip(" .;,:") for x in re.split(r";|,|\band\b", m_gee.group(1)) if x.strip()]
    m_leg = LEGAL_DESC_PAT.search(text)
    legal = (m_leg.group(1).strip() if m_leg else '')
    return {
        'date_of_document': date.group(1) if date else '',
        'grantors': grantors,
        'grantees': grantees,
        'legal_description': legal[:800]
    }

# ---------- Clauses & Surface-only ----------
CLAUSE_PATTERNS = [
    r'surface\s+only', r'surface\s+estate\s+only', r'convey(?:ance)?\s+of\s+(?:the\s+)?surface\s+estate',
    r'reserving', r'hereby reserve', r'less and except', r'save and except',
    r'reserves', r'reserving unto', r'subject to', r'burdened by', r'deducted from',
    r'excepting all oil, gas and other minerals',
    r'excepting oil, gas, and other hydrocarbons',
    r'an undivided [\d/\.]+\s*(?:%|percent)?\s+of (?:his|her|its) right, title, and interest',
]

def is_surface_only(text: str) -> bool:
    return bool(
        re.search(r'\b(surface\s+only|surface\s+estate\s+only|convey(?:ance)?\s+of\s+the?\s*surface\s+estate)\b', text, re.I) or
        re.search(r'\bexcluding\s+all\s+oil,?\s*gas.*\b', text, re.I) or
        re.search(r'\breserving\s+all\s+oil,?\s*gas.*\b', text, re.I)
    )

def find_clauses(text: str, extra: str=''):
    pats = CLAUSE_PATTERNS.copy()
    if extra: pats.append(re.escape(extra))
    hits = {}
    for pat in pats:
        found = re.findall(pat, text, flags=re.I)
        if found:
            hits[pat] = found
    return hits

# ---------- Grouping (batch) ----------
FIRST_PAGE_PAT = re.compile(r"(WARRANTY DEED|MINERAL DEED|QUIT ?CLAIM|ASSIGNMENT|LEASE|PATENT|RELEASE|AFFIDAVIT|EASEMENT)", re.I)
INSTR_PAT      = re.compile(r"(Instrument No\.?\s*\d+|Doc(?:ument)? #\s*\d+|Vol(?:ume)?\s*\d+\s*Pg(?:age)?\s*\d+)", re.I)
PAGINATION_PAT = re.compile(r"Page\s+\d+\s+of\s+\d+", re.I)
INDEX_HDR_PAT  = re.compile(r"(Grantor|Grantee|Instr|Instrument|Vol\.?|Pg|Recorded|Date)", re.I)

def quick_ocr_for_grouping(pil: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil)
    except Exception:
        return ''

def is_index_page(text: str) -> bool:
    lines = text.splitlines()[:25]
    hits = sum(1 for ln in lines if INDEX_HDR_PAT.search(ln))
    return hits >= 2

def is_first_page(text: str) -> bool:
    return bool(FIRST_PAGE_PAT.search(text) or INSTR_PAT.search(text))

def is_last_page(text: str) -> bool:
    return bool(PAGINATION_PAT.search(text))

def group_pages_into_docs(images: List[Image.Image]):
    docs, current = [], []
    for pil in images:
        txt = quick_ocr_for_grouping(pil)
        if is_index_page(txt):
            continue
        if (not current) or is_first_page(txt):
            if current: docs.append(current)
            current = [(pil, txt)]
        else:
            current.append((pil, txt))
            if is_last_page(txt):
                docs.append(current); current = []
    if current: docs.append(current)
    return docs

def classify_instrument(text: str) -> str:
    m = FIRST_PAGE_PAT.search(text)
    return (m.group(1).upper().replace(' ', '_') if m else 'OTHER')

# ---------- Ledger Model ----------
TRACT_FALLBACK = 'TRACT-DEFAULT'
SCOPE_ALL = ('ALL_MINERALS', 'ALL_DEPTHS')

class BaseBook:
    def __init__(self, label='LEDGER'):
        self.shares = {}
        self.audit = []
        self.tract_acres = {}
        self.label = label

    def set_tract_acres(self, tract, acres):
        try:
            self.tract_acres[tract] = Fraction(int(float(acres)*1000), 1000)
        except Exception:
            pass

    def _key(self, owner, tract, scope, itype):
        return (owner, tract, scope, itype)

    def credit(self, owner, tract, scope, itype, frac, note=''):
        k = self._key(owner, tract, scope, itype)
        self.shares[k] = self.shares.get(k, Fraction(0,1)) + frac
        self.audit.append({'ledger':self.label,'action':'credit','owner':owner,'tract':tract,'scope':scope,'itype':itype,'frac':str(frac),'note':note})

    def debit(self, owner, tract, scope, itype, frac, note=''):
        k = self._key(owner, tract, scope, itype)
        self.shares[k] = self.shares.get(k, Fraction(0,1)) - frac
        self.audit.append({'ledger':self.label,'action':'debit','owner':owner,'tract':tract,'scope':scope,'itype':itype,'frac':str(frac),'note':note})

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
    s = str(text).strip().lower().replace('%',' percent')
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if m: return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(percent|%)", s)
    if m: return Fraction(int(float(m.group(1))*10000), 10000*100)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m: return Fraction(int(float(m.group(1))*10000), 10000)
    return Fraction(0,1)

# ---------- Event extraction (heuristic) ----------
def regex_events_from_text(text: str, tract: str) -> list:
    evts = []
    surface_flag = is_surface_only(text)

    # PATENT
    if re.search(r"\bPATENT\b|\bGRANT\s+FROM\s+STATE\b", text, re.I):
        m = re.search(r"(?:to|unto)\s+([A-Z][A-Za-z.,'\-\s]+)", text)
        patentee = m.group(1).strip() if m else "UNKNOWN PATENTEE"
        evts.append({'type':'ROOT_PATENT','grantee':patentee,'interest_type':'MIR','fraction':'1/1','tract':tract,'scope':SCOPE_ALL})

    # UND ACRES
    m_ac = re.search(r"an?\s+undivided\s+([\d,]+(?:\.\d+)?)\s*acre", text, re.I)
    und_acres = m_ac.group(1).replace(',', '') if m_ac else None

    # CONVEY (very basic)
    m_names = re.search(r"between\s+(.*?)\s+and\s+(.*?)[,\n]", text, re.I)
    if m_names:
        grantor = m_names.group(1).strip()
        grantee = m_names.group(2).strip()
        e = {'type':'CONVEY','grantor':grantor,'grantee':grantee,'fraction':'1/1','interest_type':'MIR','scope':SCOPE_ALL,'tract':tract,'surface_only':surface_flag}
        if und_acres: e['acre_amount'] = und_acres
        evts.append(e)

    return evts

def apply_event_to_book(book: BaseBook, e: dict):
    et = e.get('type'); itype = e.get('interest_type','MIR')
    tract = e.get('tract', TRACT_FALLBACK)
    scope = SCOPE_ALL
    frac = parse_fraction(e.get('fraction','1/1'))

    if et == 'ROOT_PATENT':
        gr = e.get('grantee','UNKNOWN')
        book.credit(gr, tract, scope, itype, frac, note='ROOT_PATENT'); return

    if et == 'CONVEY':
        grantor = e.get('grantor','UNKNOWN'); grantee = e.get('grantee','UNKNOWN')
        cur = book.available(grantor, tract, scope, itype)
        if cur <= Fraction(0,1): return
        give = cur if frac == Fraction(1,1) else cur * frac
        acre_amount = e.get('acre_amount')
        if acre_amount and tract in book.tract_acres:
            try:
                acres = Fraction(int(float(str(acre_amount))*1000), 1000)
                give_from_acres = acres / book.tract_acres[tract]
                if give_from_acres < give:
                    give = give_from_acres
            except Exception:
                pass
        book.debit(grantor, tract, scope, itype, give, note='CONVEY')
        book.credit(grantee, tract, scope, itype, give, note='CONVEY'); return

# ---------- UI ----------
tab_single, tab_batch = st.tabs(['Single Document', 'Batch Mode'])

with tab_single:
    upl = st.file_uploader('Upload (PDF/JPG/PNG/TIFF)', type=['pdf','png','jpg','jpeg','tiff'], key='upl_single')
    extra_clause = st.text_input('Additional clause/phrase (optional)', key='clause_single')

    if st.button('Process Document', key='btn_single') and upl:
        txt = extract_full_text(upl.read(), upl.type)
        if not txt: st.stop()
        st.subheader('OCR Text')
        st.text_area('Output', txt, height=220)

        meta_h = parse_metadata_heuristic(txt)
        st.subheader('Heuristic Metadata')
        st.json(meta_h)

        hits = find_clauses(txt, extra_clause)
        st.subheader('Clause Matches')
        st.json(hits)

        # Excel export
        row = {
            'filename': upl.name,
            'surface_only': bool(is_surface_only(txt)),
            'clauses_found': '; '.join([c for sub in hits.values() for c in sub]) if hits else '',
            'date_of_document': meta_h.get('date_of_document',''),
            'grantors': ', '.join(meta_h.get('grantors', [])),
            'grantees': ', '.join(meta_h.get('grantees', [])),
            'legal_description': meta_h.get('legal_description',''),
        }
        df = pd.DataFrame([row])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, index=False)
        st.download_button('Download Excel', buf.getvalue(), 'quantumdocs_single.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_single')

with tab_batch:
    st.markdown('### Seeds & Tract Acres (optional)')
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        seed_owner = st.text_input('Seed Owner/Patentee', key='seed_owner', placeholder='John Smith')
    with c2:
        seed_tract = st.text_input('Tract ID', key='seed_tract', placeholder='TRACT-DEFAULT')
    with c3:
        seed_fraction_str = st.text_input('Seed Fraction', key='seed_fraction', value='1/1')

    st.caption('If given, the mineral ledger starts with this owner on ALL_MINERALS/ALL_DEPTHS.')

    st.markdown('#### Tract Acres')
    acres_text = st.text_area('tract,acres per line', key='acres_text', height=70, placeholder='TRACT-DEFAULT,640')

    st.markdown('#### Inputs')
    clause_b = st.text_input('Batch: extra clause (optional)', key='clause_batch')
    zf = st.file_uploader('Upload ZIP', type=['zip'], key='zip_batch')
    files = st.file_uploader('Or multiple files', type=['pdf','png','jpg','jpeg','tiff'], accept_multiple_files=True, key='files_batch')

    def load_images_from_inputs(zf, files):
        images = []
        if zf:
            with tempfile.TemporaryDirectory() as td:
                zp = os.path.join(td, 'f.zip')
                with open(zp, 'wb') as f: f.write(zf.read())
                with zipfile.ZipFile(zp, 'r') as z: z.extractall(td)
                for p in sorted(glob.glob(td + '/**', recursive=True)):
                    if os.path.isfile(p) and p.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                        try: images.append(Image.open(p).convert('RGB'))
                        except: pass
                    elif os.path.isfile(p) and p.lower().endswith('.pdf') and convert_from_bytes:
                        try:
                            with open(p, 'rb') as pf:
                                images.extend(convert_from_bytes(pf.read()))
                        except: pass
        for f in (files or []):
            data = f.read()
            if f.type == 'application/pdf' and convert_from_bytes:
                try: images.extend(convert_from_bytes(data))
                except: pass
            else:
                try: images.append(Image.open(io.BytesIO(data)).convert('RGB'))
                except: pass
        return images

    if st.button('Run Batch', key='btn_batch'):
        images = load_images_from_inputs(zf, files)
        if not images:
            st.warning('No readable images found.'); st.stop()

        # Build tract acres map
        tract_acres = {}
        if acres_text:
            for line in acres_text.splitlines():
                if not line.strip(): continue
                try:
                    t,a = [x.strip() for x in line.split(',',1)]
                    tract_acres[t] = float(a)
                except: pass
        if 'TRACT-DEFAULT' not in tract_acres:
            tract_acres['TRACT-DEFAULT'] = 640.0

        st.write(f'Loaded {len(images)} pages. Grouping into documents…')
        docs = group_pages_into_docs(images)

        rows = []
        events_by_doc = []
        for pages in docs:
            all_text = '\n'.join([pytesseract.image_to_string(p[0]) for p in pages])
            meta = parse_metadata_heuristic(all_text)

            tract = 'TRACT-DEFAULT'
            evts = regex_events_from_text(all_text, tract)

            clause_hits = find_clauses(all_text, clause_b)
            rows.append({
                'instrument_type': classify_instrument(all_text),
                'date_of_document': meta.get('date_of_document',''),
                'grantors': meta.get('grantors',[]),
                'grantees': meta.get('grantees',[]),
                'legal_description': meta.get('legal_description',''),
                'clauses_found': '; '.join([c for sub in clause_hits.values() for c in sub]),
                'page_count': len(pages),
                'surface_only': is_surface_only(all_text)
            })
            try:
                d = meta.get('date_of_document','')
                dt = datetime.strptime(d, '%B %d, %Y') if d and ',' in d else datetime.max
            except Exception:
                dt = datetime.max
            events_by_doc.append({'date': dt, 'events': evts})

        df = pd.DataFrame(rows)
        st.subheader('Batch Results')
        st.dataframe(df, use_container_width=True)
        bbuf = io.BytesIO()
        with pd.ExcelWriter(bbuf, engine='openpyxl') as w:
            df.to_excel(w, index=False)
        st.download_button('Download Batch Excel', bbuf.getvalue(), 'quantumdocs_batch.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_batch')

        # Ledgers
        bookM = MineralBook(label='MINERAL')
        bookS = SurfaceBook(label='SURFACE')
        # Set tract acres
        for t,a in tract_acres.items():
            bookM.set_tract_acres(t, a); bookS.set_tract_acres(t, a)
        # Seed
        if seed_owner and seed_fraction_str:
            try:
                seed_frac = parse_fraction(seed_fraction_str)
            except Exception:
                seed_frac = Fraction(1,1)
            tseed = seed_tract.strip() or 'TRACT-DEFAULT'
            bookM.credit(seed_owner.strip(), tseed, SCOPE_ALL, 'MIR', seed_frac, note='SEEDED')
            bookS.credit(seed_owner.strip(), tseed, SCOPE_ALL, 'SURFACE', seed_frac, note='SEEDED')

        # Apply events
        for pkg in sorted(events_by_doc, key=lambda x: x['date']):
            for e in pkg['events']:
                if e.get('type') == 'ROOT_PATENT':
                    apply_event_to_book(bookM, e)
                    eS = e.copy(); eS['interest_type'] = 'SURFACE'
                    apply_event_to_book(bookS, eS)
                elif e.get('surface_only'):
                    e2 = e.copy(); e2['interest_type'] = 'SURFACE'
                    apply_event_to_book(bookS, e2)
                else:
                    apply_event_to_book(bookM, e)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Mineral Ledger')
            dfM = bookM.to_dataframe()
            if not dfM.empty:
                st.dataframe(dfM, use_container_width=True)
                obuf = io.BytesIO()
                with pd.ExcelWriter(obuf, engine='openpyxl') as w:
                    dfM.to_excel(w, index=False)
                st.download_button('Download Mineral Ledger (Excel)', obuf.getvalue(), 'quantumdocs_mineral_ledger.xlsx',
                                   'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_mineral')
            else:
                st.info('No mineral ownership changes detected.')

        with col2:
            st.subheader('Surface Ledger')
            dfS = bookS.to_dataframe()
            if not dfS.empty:
                st.dataframe(dfS, use_container_width=True)
                sbuf = io.BytesIO()
                with pd.ExcelWriter(sbuf, engine='openpyxl') as w:
                    dfS.to_excel(w, index=False)
                st.download_button('Download Surface Ledger (Excel)', sbuf.getvalue(), 'quantumdocs_surface_ledger.xlsx',
                                   'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_surface')
            else:
                st.info('No surface ownership changes detected.')

        st.subheader('Audit Trail')
        audit_all = pd.DataFrame(bookM.audit + bookS.audit)
        if not audit_all.empty:
            st.dataframe(audit_all, use_container_width=True)
            abuf = io.BytesIO()
            with pd.ExcelWriter(abuf, engine='openpyxl') as w:
                audit_all.to_excel(w, index=False)
            st.download_button('Download Audit (Excel)', abuf.getvalue(), 'quantumdocs_audit.xlsx',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_audit')
            st.download_button('Download Audit (JSON)', json.dumps(bookM.audit + bookS.audit, indent=2).encode('utf-8'),
                               'quantumdocs_audit.json', 'application/json', key='dl_audit_json')
        else:
            st.info('No audit entries recorded.')

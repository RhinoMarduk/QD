
import streamlit as st, os, subprocess, sys, json, io
st.set_page_config(page_title="QuantumDocs Diag", layout="centered")
st.title("🔎 QuantumDocs — Environment Diagnostics")
st.caption("Minimal app to verify Streamlit Cloud dependencies.")

def which(cmd):
    try:
        p = subprocess.run(["which", cmd], capture_output=True, text=True)
        return p.stdout.strip() or "(not found)"
    except Exception as e:
        return f"error: {e}"

def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or p.stderr).strip()
        return out[:800]
    except Exception as e:
        return f"error: {e}"

st.subheader("System Binaries")
tesseract_path = which("tesseract")
pdftoppm_path  = which("pdftoppm")
st.code(f"tesseract: {tesseract_path}\npdftoppm: {pdftoppm_path}")

st.subheader("Python Packages")
report = {}
for mod in ["pytesseract", "pdf2image", "PIL", "pandas", "openpyxl", "streamlit"]:
    try:
        m = __import__(mod if mod != "PIL" else "PIL")
        ver = getattr(m, "__version__", "OK")
        report[mod] = f"OK ({ver})"
    except Exception as e:
        report[mod] = f"IMPORT FAIL: {e}"
st.json(report)

st.subheader("Tesseract Version")
st.code(run(["tesseract", "--version"]))

st.subheader("Poppler (pdftoppm) Version")
st.code(run(["pdftoppm", "-v"]))

st.subheader("Quick OCR Test (Image only)")
upl = st.file_uploader("Upload PNG/JPG (small)", type=["png","jpg","jpeg"])
if upl and st.button("Run OCR"):
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(upl).convert("RGB")
        txt = pytesseract.image_to_string(img)
        st.text_area("OCR Output", txt, height=200)
    except Exception as e:
        st.error(f"OCR error: {e}")

st.divider()
st.caption("If this page loads and shows versions, your Cloud env is fine. If binaries are '(not found)', packages.txt isn't being applied or repo layout isn't root-level.")

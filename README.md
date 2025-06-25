
## 📘 Automated Metadata Generation Web App

This project is a **Streamlit-based web application** for extracting metadata, summarizing, and analyzing documents (PDF, DOCX, image-based, or plain text) using NLP techniques.

---

### 🧰 Features

* 📄 **Document parsing** (PDF, DOCX, image OCR)
* 🧠 **NLP Analysis** (keywords, entities, sentiment, summary)
* 📊 **Visualization** (word clouds, topic charts)
* 📝 **Readable summaries** and topic extraction
* 🌐 **Streamlit interface** for easy interaction

---

## 🚀 Getting Started

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/t0bleronee/metadata_extraction
cd automated-metadata-generator
```

---

### 📦 2. Create and Activate a Virtual Environment (Windows)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # For PowerShell
# OR
venv\Scripts\activate.bat    # For CMD
```

---

### 📥 3. Install Required Python Dependencies

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, use this:

```bash
pip install streamlit pandas plotly wordcloud matplotlib \
PyPDF2 pdfplumber python-docx pytesseract Pillow pymupdf textblob
```
```bash
Download spaCy models: `python -m spacy download en_core_web_sm`
```
---

### 🧠 4. Install Tesseract-OCR (for image OCR)

* 📥 Download from: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* 🔧 After install, set path in your code if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## ▶️ Running the App

### ✅ Make sure the virtual environment is activated, then run:

```bash
streamlit run app.py
```

Or (for PowerShell path issues):

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
automated-metadata-generator/
│
├── app.py                         # Streamlit main app
├── requirements.txt              # Package dependencies
├── README.md                     # Project guide
```
 document_processor     # PDF, DOCX, OCR logic
 nlp_analyzer           # NLP logic
 metadata_generator     # Combines all metadata outputs
`
---

## 🌐 Live App
## 🔗 Try it here:
👉 https://metadataextraction-m7ciiquhepcup25ftappdpr.streamlit.app/

---


## Features
- Document processing (PDF, DOCX, TXT)
- OCR for scanned documents
- Semantic content analysis
- Structured metadata generation
- Web interface

---

## Tech Stack
- Python 3.8+
- Streamlit/Flask
- spaCy, NLTK
- PyPDF2, python-docx
- Tesseract OCR

---
## Supports:

* `.pdf`, `.docx`, `.txt`
* `.png`, `.jpg` (OCR)
* Copy-pasted plain text

---
## 🐞 Troubleshooting

* **`streamlit: command not found`** → Use `python -m streamlit run app.py`
* **`No module named ...`** → Reinstall missing modules inside the `venv`
* **OCR not working** → Ensure Tesseract is installed and path is set correctly

---

## 📌 License

MIT License. Free to use, modify, and share.

---

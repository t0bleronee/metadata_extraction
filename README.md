# 🤖 Intelligent Document Metadata Extraction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

**An AI-powered document intelligence platform that transforms unstructured documents into actionable metadata using advanced NLP and machine learning techniques.**

</div>

---

## 🎯 **Project Vision**

In today's data-driven world, organizations struggle with vast amounts of unstructured document data. Our **Intelligent Document Metadata Extraction System** addresses this challenge by automatically transforming documents into structured, searchable metadata using cutting-edge AI techniques.

### **The Problem We Solve**
- **Information Overload**: Manual document processing is time-consuming and error-prone
- **Knowledge Silos**: Important information locked in unstructured formats
- **Compliance Challenges**: Difficulty in document classification and content analysis
- **Search Inefficiency**: Poor discoverability of relevant documents

### **Our Solution**
A comprehensive AI-powered system that intelligently processes, analyzes, and extracts meaningful metadata from diverse document types, enabling organizations to unlock the value of their document repositories.

---

## 🏗️ **System Architecture & Approach**

### **Multi-Modal Processing Pipeline**

Our system employs a sophisticated three-stage architecture designed for scalability and accuracy:

```mermaid
graph TD
    A[📄 Document Input] --> B[⚙️ Document Processor]
    B --> C[ NLP Analyzer]
    C --> D[ Metadata Generator]
    D --> E[ Structured Output]

    %% Document Processor Components
    B --> B1[ PDF Parser]
    B --> B2[ OCR Engine]
    B --> B3[ Text Extractor]

    %% NLP Analyzer Components
    C --> C1[ Entity Recognition]
    C --> C2[ Topic Modeling]
    C --> C3[ Sentiment Analysis]
    C --> C4[ Summarization]

    %% Metadata Generator Outputs
    D --> D1[ JSON-LD Export]
    D --> D2[ XML Export]
    D --> D3[ Database Storage]
```

### **1. 📄 Document Processing Layer**
- **Multi-format Support**: Native handling of PDF, DOCX, TXT, and image-based documents
- **Advanced OCR**: Tesseract-powered optical character recognition for scanned documents
- **Error Handling**: Robust exception handling for corrupted or unusual file formats
  
---

### **2. 🧠 Natural Language Processing Engine**

- **Named Entity Recognition**: Identifies domain-specific entities such as emails, URLs, dates, currencies, percentages, organizations, and phone numbers using regex-based techniques.
- **Topic Modeling**: Extracts document themes using keyword-based relevance scoring and simple clustering without requiring training data.
- **Sentiment Analysis**: Placeholder structure for integrating multi-dimensional sentiment scoring (positive, negative, neutral).
- **Automatic Summarization**: Performs extractive summarization using sentence importance heuristics (position, length, keywords).
- **Keyword Extraction**: Identifies relevant keywords using frequency-based scoring with stopword filtering.
- **Language Detection**: Stubbed for automatic language identification; ready for integration with external language models.
- **Document Structure Analysis**: Detects layout patterns including titles, headings, lists, and paragraph segmentation.
- **Text Preprocessing**: Cleans and normalizes raw input (removing noise, fixing Unicode, trimming whitespace) for consistent NLP output.

---

### **3. 🔧 Metadata Generation Framework**

- **Structured Output**  
  Produces standardized metadata formats such as **JSON-LD** and **Dublin Core**, ensuring compatibility with semantic web standards and metadata repositories.
- **Confidence Scoring**  
  Each extracted metadata field is accompanied by a **machine learning-based confidence score**, helping users assess the reliability of the information.
- **Custom Taxonomies**  
  Supports **domain-specific classification schemes** for tagging and categorizing content based on organizational or industry standards.
- **Temporal Analysis**  
  Performs **document aging and relevance scoring** by analyzing creation, modification, and access timestamps.

---
### Features

*  **Document parsing** (PDF, DOCX, image OCR)
*  **NLP Analysis** (keywords, entities, sentiment, summary)
*  **Visualization** (word clouds, topic charts)
*  **Readable summaries** and topic extraction
*  **Streamlit interface** for easy interaction

---

## 🛠️ **Technical Stack & Implementation**

### **Core Technologies**
- **Backend**: Python 3.8+ with async/await support
- **NLP Framework**: spaCy, NLTK, Transformers (HuggingFace)
- **ML Libraries**: scikit-learn, TensorFlow/PyTorch for custom models
- **Document Processing**: PyMuPDF, python-docx, pytesseract
- **Web Framework**: Streamlit for rapid prototyping
  
---

## 📋 **Installation & Setup**

### **Prerequisites**
- Python 3.8 or higher
- Tesseract OCR engine
- Git

### **Quick Start**

```bash
# Clone the repository
git clone https://github.com/t0bleronee/metadata_extraction.git
cd metadata_extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg  # Optional: Better accuracy

# Install Tesseract-OCR (for image OCR)
* Download from: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* After install, set path in your code if needed:
python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Run the Application
streamlit run app.py
```
---

## 🌐 Live App
### 🔗 Try it here:
👉 https://metadataextraction-m7ciiquhepcup25ftappdpr.streamlit.app/

---
##  **Usage Examples**

### **Basic Usage**

```python
from metadata_extractor import MetadataExtractor

# Initialize the system
extractor = MetadataExtractor()

# Process a single document
metadata = extractor.process_document("path/to/document.pdf")
print(f"Extracted {len(metadata.keywords)} keywords")
print(f"Document sentiment: {metadata.sentiment}")

# Batch processing
results = extractor.process_directory("./documents/")
extractor.generate_report(results, "analysis_report.html")
```

```python
# Custom entity extraction
custom_entities = ["Product Names", "Project Codes", "Department Names"]
extractor.add_custom_entities(custom_entities)

# Domain-specific processing
extractor.set_domain("legal")  # Optimizes for legal documents

```

### **API Usage**

```python
import requests

# REST API endpoint
response = requests.post(
    "http://localhost:8501/api/extract",
    files={"document": open("document.pdf", "rb")},
    data={"include_summary": True, "language": "en"}
)

metadata = response.json()
```

---


## 🎨 **User Interface & Experience**

### **Web Interface Features**
- **Drag & Drop Upload**: Intuitive file upload with progress tracking
- **Real-time Processing**: Live updates during document analysis
- **Interactive Visualizations**: Word clouds, topic distributions, sentiment trends
- **Export Options**: Multiple format support (JSON, CSV, XML, PDF report)
- **Responsive Design**: Works seamlessly on desktop and mobile devices


### **Jupyter Notebook Integration**
- **Interactive Analysis**: Step-by-step document processing
- **Visualization Widgets**: Interactive charts and graphs
- **Experimentation**: Easy testing of different parameters
- **Export Options**: Multiple format support (JSON, CSV, XML, PDF report)


---

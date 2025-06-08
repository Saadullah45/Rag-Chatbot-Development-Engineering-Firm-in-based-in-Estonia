#  Engineering Document Chatbot — AI-Powered Technical Assistant

An intelligent, domain-specific chatbot built for an engineering firm to retrieve and answer technical questions from thousands of internal documents (PDFs, Word files, images, and product manuals) stored on Google Drive. This project leverages a **Retrieval-Augmented Generation (RAG)** pipeline with multilingual capabilities to provide accurate, fast, and context-aware responses.

---

##  Project Objective

Engineers often need quick answers from complex documentation. This chatbot solves that by enabling **natural language interaction** with unstructured content, saving time, reducing manual search, and increasing productivity in technical environments.

---

##  Data Source

- **Document Types**: PDFs, DOCX files, images (scanned diagrams and manuals), technical product documentation  
- **Storage**: All files were originally stored in a Google Drive folder

---

##  Pipeline Overview

### 1.  Google Drive Integration
- Connected a notebook to Google Drive using `Google Drive API` + `PyDrive`
- Iterated through thousands of files, downloading and identifying type-specific extraction logic

### 2.  Text Extraction & Cleaning

**Text Extraction:**
- PDFs: Extracted with `PyMuPDF`
- Word files: Extracted with `python-docx`
- Images: OCR using `Tesseract`

**Cleaning:**
- Removed irrelevant noise (e.g., page numbers, repeated headers)
- Normalized formatting and encoding issues

 Output: Cleaned text saved back to a structured directory in Google Drive

### 3.  Text Chunking

- Long documents split into overlapping chunks
- Token-aware logic to stay within model context limits

### 4.  Embedding Creation

```python
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
```

- Embeddings generated for each chunk
- Stored in **Pinecone** vector database for fast semantic retrieval

### 5.  Retrieval & Generation (RAG)

**Semantic Search:**
- Pinecone search with `similarity_threshold = 0.75`
- Retrieved top relevant chunks per query

**Answer Generation:**
- Used `mistral-7b-instruct-v0.3`
- Passed query + context to the model for accurate response

---

## Tech Stack

| Task               | Technology/Library                       |
|--------------------|------------------------------------------|
| File Storage       | Google Drive                             |
| Notebook Environment| Kaggle                                  |
| Text Extraction    | PyMuPDF, python-docx, Tesseract OCR      |
| Text Cleaning      | Regex, CleanText, LangDetect             |
| Chunking           | Custom tokenizer-aware sliding window    |
| Embeddings         | intfloat/multilingual-e5-large-instruct  |
| Vector Storage     | Pinecone                                 |
| LLM for QA         | mistral-7b-instruct-v0.3                 |

---

##  Example Use Case

**Q:** What features make Yuasa NP batteries reliable and versatile?
**A:** _[Chatbot extracts the relevant paragraph from the product manual and provides the exact figure with explanation.]_

---

##  Features

-  Understands technical language from diverse formats  
-  Multilingual question support  
-  Semantic search across thousands of documents  
-  Tailored for engineering use cases  
-  Human-like, accurate responses  

---

##  Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/engineering-doc-chatbot.git

# Setup Drive API credentials and mount Drive in your notebook

# Run preprocessing to extract and clean documents

# Generate embeddings and upload to Pinecone

# Query via notebook or build a frontend (Streamlit/Gradio optional)
```

---

##  Notes

- Ensure `mistral-7b` weights are loaded using Hugging Face or your preferred inference server
- Optimize similarity threshold based on your domain's recall/precision trade-off
- Easily extend to support continuous ingestion via scheduled updates

---

##  License

This project is licensed under the [MIT License](LICENSE).


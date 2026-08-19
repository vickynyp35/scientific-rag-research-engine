# 🔬 Scientific RAG Research Engine

An AI-powered research assistant that allows users to upload multiple scientific research papers and ask questions based on their content.

The system uses **Hybrid Retrieval-Augmented Generation (RAG)** with FAISS, BM25, and Cross-Encoder reranking to retrieve relevant research content and generate citation-based answers using Ollama.

## 🚀 Features

* 📄 Upload multiple scientific PDF research papers
* ✂️ Automatic text extraction and chunking
* 🔍 FAISS semantic search
* 🔤 BM25 keyword search
* 🔀 Hybrid retrieval combining semantic and keyword search
* 🧠 Cross-Encoder reranking
* 🎯 Retrieval relevance scoring
* 🤖 AI-powered question answering
* 📚 Source-based answers with file and page citations
* ⚖️ Multi-paper comparison
* 🤝 Identify agreements between papers
* ⚡ Identify contradictions between papers
* 🔎 Identify potential research gaps
* 🛡️ Source-grounded answers to reduce unsupported responses

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **FAISS**
* **BM25**
* **Sentence Transformers**
* **Cross-Encoder**
* **Ollama**
* **PyMuPDF**
* **Requests**

## 🧠 RAG Pipeline

```text
Scientific Research Papers
          ↓
      PDF Upload
          ↓
   Text Extraction
          ↓
      Chunking
          ↓
 ┌─────────────────────┐
 │  FAISS Semantic     │
 │  Search             │
 └─────────────────────┘
          +
 ┌─────────────────────┐
 │  BM25 Keyword       │
 │  Search             │
 └─────────────────────┘
          ↓
    Hybrid Retrieval
          ↓
 Cross-Encoder Reranking
          ↓
   Relevance Filtering
          ↓
     Context Building
          ↓
      Ollama LLM
          ↓
 Citation-Based Answer
```

## 📊 Paper Comparison

The system can compare multiple uploaded research papers and analyze:

* 🤝 Agreements
* ⚡ Contradictions
* 🔎 Research gaps

The comparison is performed using only the uploaded research papers.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/vickynyp35/scientific-rag-research-engine.git
cd scientific-rag-research-engine
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Make sure Ollama is installed and running on your computer.

The application currently uses:

```text
llama3.2:3b
```

Pull the model if necessary:

```bash
ollama pull llama3.2:3b
```

### 5. Start the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
scientific-rag-research-engine/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Example Use Cases

This system can be used for:

* 📚 Research paper analysis
* 🔬 Scientific literature exploration
* ⚖️ Comparing multiple papers
* 🔎 Finding research gaps
* 💬 Question answering over research documents
* 📖 Academic research assistance

## 🔮 Future Improvements

* OCR support for scanned PDFs
* Persistent vector database
* Conversation history
* Support for additional LLM providers
* Improved citation verification
* Advanced research-paper summarization
* Cloud deployment

## 👨‍💻 Project

**Scientific RAG Research Engine**

Built using Python, Streamlit, FAISS, BM25, Sentence Transformers, Cross-Encoder Reranking, and Ollama.

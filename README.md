# 🔬 Scientific RAG Research Engine

A Multi-Paper Hybrid Retrieval-Augmented Generation (RAG) system for querying and comparing scientific research papers.

## 🚀 Live Demo

https://scientific-rag-research-engine-xmzmmntq79usfdbcfazvav.streamlit.app/

## 📌 Overview

Scientific RAG Research Engine allows users to upload multiple research papers and ask questions based only on the uploaded documents.

The system combines semantic search, keyword search, hybrid retrieval, and Cross-Encoder reranking to retrieve relevant research content before generating answers using Gemini.

## ✨ Features

- 📄 Multi-PDF research paper upload
- ✂️ Text extraction and chunking
- 🔢 Sentence Transformer embeddings
- 🔍 FAISS semantic search
- 🔤 BM25 keyword search
- 🔀 Hybrid retrieval
- 🧠 Cross-Encoder reranking
- 🤖 Gemini-powered answer generation
- 📚 Source and page citations
- 📊 Multi-paper comparison
- 🤝 Agreement detection
- ⚡ Contradiction detection
- 🔎 Research gap detection
- ☁️ Streamlit Cloud deployment

## 🏗️ System Architecture

Research PDFs
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer
     ↓
FAISS Vector Search
     +
BM25 Keyword Search
     ↓
Hybrid Retrieval
     ↓
Cross-Encoder Reranking
     ↓
Top Relevant Sources
     ↓
Gemini
     ↓
Citation-Based Answer

🛠️ Technologies Used
Python
Streamlit
PyMuPDF
FAISS
BM25
Sentence Transformers
Cross-Encoder
Gemini API
Requests

🔬 How It Works
1. Document Processing

Research papers are uploaded as PDF files and their text is extracted using PyMuPDF.

2. Chunking

Extracted text is divided into smaller overlapping chunks for efficient retrieval.

3. Semantic Retrieval

Sentence Transformer embeddings are generated for each chunk and stored in a FAISS vector index.

4. Keyword Retrieval

BM25 is used to identify chunks containing important query terms.

5. Hybrid Retrieval

Semantic and keyword scores are combined to improve retrieval quality.

6. Reranking

A Cross-Encoder reranks the retrieved chunks based on their relevance to the user's question.

7. Answer Generation

The most relevant research content is sent to Gemini with strict instructions to answer only from the provided papers.

📊 Paper Comparison

The system supports:

Agreements between papers
Contradictions and differences
Research gaps
Limitations and future research directions
💬 Example Questions
What are the main findings of these research papers?


What are the limitations mentioned in the research papers?


What methods were used in the papers?


What are the similarities between the research papers?


What research gaps are identified?
🔐 API Key

The Gemini API key is stored securely using Streamlit Secrets.

Do not expose API keys in source code or GitHub repositories.

▶️ Run Locally

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd scientific-rag-research-engine

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
📁 Project Structure
scientific-rag-research-engine/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🎯 Project Goal

The goal of this project is to provide a reliable research assistant that helps users retrieve relevant information from multiple scientific papers while maintaining source-based answers.

👨‍💻 Author

Vigneshwaran B,
Rahul S,
Sivasakthi S,
Sanjay M,
Praveen P
(B.Tech Artificial Intelligence and Data Science)

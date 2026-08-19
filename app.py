import streamlit as st
import pymupdf
import faiss
import requests
import re
import time

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Scientific RAG Research Engine",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Scientific RAG Research Engine")

st.caption(
    "Multi-Paper Hybrid RAG with "
    "FAISS + BM25 + Cross-Encoder Reranking"
)


# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

OLLAMA_TAGS_URL = (
    "http://localhost:11434/api/tags"
)

OLLAMA_MODEL = "llama3.2:3b"

TOP_K_FAISS = 10
TOP_K_HYBRID = 10
TOP_K_RERANK = 8
TOP_K_FINAL = 5

MAX_CONTEXT_CHARS = 7000

MIN_HYBRID_SCORE = 0.18
MIN_RELEVANCE_SCORE = 0.25

FALLBACK_ANSWER = (
    "I could not find this information "
    "in the provided research papers."
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# =========================================================
# LOAD RERANKER
# =========================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL_NAME
    )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    step = chunk_size - overlap

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# =========================================================
# TOKENIZER
# =========================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# =========================================================
# SCORE NORMALIZATION
# =========================================================

def min_max_normalize(values):

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:

        return [
            0.5
            for _ in values
        ]

    return [
        (value - minimum) /
        (maximum - minimum)
        for value in values
    ]


# =========================================================
# OLLAMA HEALTH CHECK
# =========================================================

def check_ollama():

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=10
        )

        if response.status_code != 200:

            return False, (
                f"Ollama server returned "
                f"HTTP {response.status_code}"
            )

        data = response.json()

        models = [
            item.get("name", "")
            for item in data.get("models", [])
        ]

        if OLLAMA_MODEL not in models:

            return False, (
                f"Model '{OLLAMA_MODEL}' "
                f"is not available."
            )

        return True, "Ollama is ready."

    except requests.exceptions.ConnectionError:

        return False, (
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return False, (
            "Ollama health check timed out."
        )

    except Exception as e:

        return False, str(e)


# =========================================================
# GENERATE ANSWER WITH OLLAMA
# =========================================================

def generate_answer(prompt):

    payload = {

        "model": OLLAMA_MODEL,

        "prompt": prompt,

        "stream": False,

        "options": {

            "temperature": 0.0,

            "num_predict": 500,

            "num_ctx": 4096

        }

    }

    max_attempts = 2

    for attempt in range(max_attempts):

        try:

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=180
            )

            if response.status_code == 200:

                try:

                    result = response.json()

                except ValueError:

                    return None, (
                        "Ollama returned invalid JSON."
                    )

                answer = result.get(
                    "response",
                    ""
                )

                if not answer.strip():

                    return None, (
                        "Ollama returned an empty answer."
                    )

                return answer.strip(), None

            if response.status_code >= 500:

                if attempt < max_attempts - 1:

                    time.sleep(2)

                    continue

                return None, (
                    f"Ollama returned HTTP "
                    f"{response.status_code}\n\n"
                    f"{response.text}"
                )

            return None, (
                f"Ollama returned HTTP "
                f"{response.status_code}\n\n"
                f"{response.text}"
            )

        except requests.exceptions.ConnectionError:

            return None, (
                "Could not connect to Ollama."
            )

        except requests.exceptions.Timeout:

            return None, (
                "Ollama request timed out."
            )

        except Exception as e:

            return None, (
                f"Unexpected Ollama error: {e}"
            )

    return None, "Unknown Ollama error."


# =========================================================
# BUILD CONTEXT FOR QUESTION ANSWERING
# =========================================================

def build_context(
    final_results,
    pages,
    chunks,
    max_chars=MAX_CONTEXT_CHARS
):

    context_parts = []

    total_chars = 0

    used_pages = set()

    for number, item in enumerate(
        final_results,
        start=1
    ):

        idx = item["idx"]

        source = pages[idx]

        page_key = (
            source["source"],
            source["page"]
        )

        if page_key in used_pages:
            continue

        content = chunks[idx]

        remaining = (
            max_chars - total_chars
        )

        if remaining <= 0:
            break

        if len(content) > remaining:

            content = content[:remaining]

        context_parts.append(
            f"""
SOURCE {number}
File: {source['source']}
Page: {source['page']}

Content:
{content}
"""
        )

        used_pages.add(page_key)

        total_chars += len(content)

    return "\n".join(context_parts)


# =========================================================
# BUILD PAPER CONTEXT
# =========================================================

def build_paper_context(
    pages,
    chunks,
    max_chars=3500
):

    papers = {}

    for idx, source in enumerate(pages):

        filename = source["source"]

        if filename not in papers:
            papers[filename] = []

        papers[filename].append(
            chunks[idx]
        )

    context_parts = []

    for filename, paper_chunks in papers.items():

        combined = "\n".join(
            paper_chunks
        )

        combined = combined[:max_chars]

        context_parts.append(
            f"""
PAPER:
{filename}

CONTENT:
{combined}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload scientific research papers",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_files:

    all_pages = []

    processed_files = 0

    failed_files = []


    # =====================================================
    # READ PDF FILES
    # =====================================================

    with st.spinner(
        "📄 Processing research papers..."
    ):

        for uploaded_file in uploaded_files:

            try:

                pdf_bytes = uploaded_file.read()

                doc = pymupdf.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )

                file_has_text = False

                for page_number, page in enumerate(
                    doc,
                    start=1
                ):

                    text = page.get_text()

                    if not text.strip():
                        continue

                    file_has_text = True

                    page_chunks = chunk_text(
                        text
                    )

                    for chunk in page_chunks:

                        all_pages.append(
                            {
                                "text": chunk,
                                "page": page_number,
                                "source": uploaded_file.name
                            }
                        )

                doc.close()

                if file_has_text:

                    processed_files += 1

                else:

                    failed_files.append(
                        f"{uploaded_file.name} "
                        f"(no readable text)"
                    )

            except Exception as e:

                failed_files.append(
                    f"{uploaded_file.name}: {e}"
                )


    # =====================================================
    # STORE DATA
    # =====================================================

    pages = all_pages

    chunks = [
        item["text"]
        for item in pages
    ]


    # =====================================================
    # EMPTY CHECK
    # =====================================================

    if not chunks:

        st.error(
            "❌ No readable text was found "
            "in the uploaded PDF files."
        )

        st.info(
            "Scanned PDFs may require OCR."
        )

        st.stop()


    # =====================================================
    # STATUS
    # =====================================================

    st.success(
        f"✅ {processed_files} PDF(s) "
        f"processed successfully!"
    )

    for failed in failed_files:

        st.warning(
            f"⚠️ {failed}"
        )

    st.write(
        f"### 📄 Total Chunks: {len(chunks)}"
    )


    # =====================================================
    # LOAD AI MODELS
    # =====================================================

    with st.spinner(
        "🤖 Loading AI models..."
    ):

        model = load_embedding_model()

        reranker = load_reranker()


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    with st.spinner(
        "🔢 Creating document embeddings..."
    ):

        embeddings = model.encode(
            chunks,
            show_progress_bar=False,
            normalize_embeddings=True
        )

    embeddings = embeddings.astype(
        "float32"
    )


    # =====================================================
    # CREATE FAISS INDEX
    # =====================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    st.success(
        "✅ FAISS vector database ready!"
    )

    st.write(
        f"### 🔍 Vectors Stored: "
        f"{index.ntotal}"
    )


    # =====================================================
    # CREATE BM25
    # =====================================================

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    st.success(
        "✅ BM25 keyword search ready!"
    )


    # =====================================================
    # PAPER COMPARISON
    # =====================================================

    if len(uploaded_files) >= 2:

        st.divider()

        st.subheader(
            "📊 Paper Comparison"
        )

        st.caption(
            "Compare the uploaded research papers "
            "for agreements, contradictions, "
            "and research gaps."
        )

        comparison_type = st.selectbox(
            "Choose comparison type:",
            [
                "🤝 Agreements",
                "⚡ Contradictions",
                "🔎 Research Gaps"
            ],
            key="comparison_type"
        )

        compare_button = st.button(
            "🔬 Analyze Papers",
            key="compare_papers"
        )

        if compare_button:

            paper_context = build_paper_context(
                pages,
                chunks
            )

            if comparison_type == "🤝 Agreements":

                comparison_instruction = """
Identify the main points where the research
papers agree with each other.

For each agreement:
- Explain the shared finding.
- Mention the relevant paper names.
- Mention page numbers when supported.
- Do not invent information.
"""

            elif comparison_type == "⚡ Contradictions":

                comparison_instruction = """
Identify meaningful contradictions or
differences between the research papers.

For each contradiction:
- Explain what Paper A reports.
- Explain what Paper B reports.
- Clearly state the difference.
- Mention paper names.
- Mention page numbers when supported.
- Do not assume that two different methods
  automatically mean contradiction.
"""

            else:

                comparison_instruction = """
Identify research gaps based ONLY on the
provided research papers.

Look for:
- Missing areas of research.
- Limitations mentioned by the papers.
- Unanswered questions.
- Areas requiring future investigation.

Do not invent research gaps that are not
reasonably supported by the provided papers.
"""

            comparison_prompt = f"""
You are a scientific research comparison assistant.

Use ONLY the research papers provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not guess.
4. Do not fabricate citations.
5. Compare only the provided papers.
6. Clearly identify the paper names.
7. Keep the analysis structured and concise.

TASK:

{comparison_instruction}

RESEARCH PAPERS:

{paper_context}

ANALYSIS:
"""

            with st.spinner(
                "🧠 AI is comparing the research papers..."
            ):

                ollama_ready, ollama_message = (
                    check_ollama()
                )

            if not ollama_ready:

                st.error(
                    f"❌ {ollama_message}"
                )

            else:

                with st.spinner(
                    "🤖 Generating comparison..."
                ):

                    comparison_answer, comparison_error = (
                        generate_answer(
                            comparison_prompt
                        )
                    )

                if comparison_answer:

                    st.subheader(
                        "📊 Comparison Result"
                    )

                    st.write(
                        comparison_answer
                    )

                else:

                    st.error(
                        "❌ Comparison analysis failed."
                    )

                    st.code(
                        comparison_error
                        if comparison_error
                        else "Unknown Ollama error.",
                        language="text"
                    )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    question = st.text_input(
        "Enter your question about the research papers:",
        key="research_question"
    )


    # =====================================================
    # SEARCH
    # =====================================================

    if question.strip():

        clean_question = question.strip()


        # =================================================
        # QUESTION VALIDATION
        # =================================================

        if len(clean_question) < 3:

            st.warning(
                "⚠️ Please enter a meaningful question."
            )

            st.stop()


        # =================================================
        # HYBRID SEARCH
        # =================================================

        with st.spinner(
            "🔍 Running hybrid search..."
        ):

            # ---------------------------------------------
            # QUERY EMBEDDING
            # ---------------------------------------------

            question_embedding = model.encode(
                [clean_question],
                normalize_embeddings=True
            ).astype(
                "float32"
            )


            # ---------------------------------------------
            # FAISS SEARCH
            # ---------------------------------------------

            faiss_k = min(
                TOP_K_FAISS,
                len(chunks)
            )

            faiss_raw, faiss_indices = (
                index.search(
                    question_embedding,
                    faiss_k
                )
            )


            # ---------------------------------------------
            # BM25 SEARCH
            # ---------------------------------------------

            tokenized_question = tokenize(
                clean_question
            )

            bm25_scores_raw = (
                bm25.get_scores(
                    tokenized_question
                )
            )


            # ---------------------------------------------
            # NORMALIZE BM25
            # ---------------------------------------------

            bm25_normalized = (
                min_max_normalize(
                    list(bm25_scores_raw)
                )
            )


            # ---------------------------------------------
            # FAISS SCORES
            # ---------------------------------------------

            faiss_scores = {}

            for rank, idx in enumerate(
                faiss_indices[0]
            ):

                idx = int(idx)

                if idx < 0:
                    continue

                score = float(
                    faiss_raw[0][rank]
                )

                faiss_scores[idx] = score


            # ---------------------------------------------
            # HYBRID SCORES
            # ---------------------------------------------

            hybrid_scores = {}

            for idx in range(
                len(chunks)
            ):

                semantic_score = (
                    faiss_scores.get(
                        idx,
                        0.0
                    )
                )

                keyword_score = float(
                    bm25_normalized[idx]
                )

                hybrid_score = (
                    0.60 * semantic_score
                    +
                    0.40 * keyword_score
                )

                hybrid_scores[idx] = (
                    hybrid_score
                )


            # ---------------------------------------------
            # SORT HYBRID
            # ---------------------------------------------

            sorted_hybrid = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            candidate_count = min(
                TOP_K_HYBRID,
                len(sorted_hybrid)
            )

            candidate_indices = [
                item[0]
                for item in sorted_hybrid[
                    :candidate_count
                ]
            ]

            candidate_indices = candidate_indices[
                :TOP_K_RERANK
            ]


        # =================================================
        # CROSS ENCODER RERANKING
        # =================================================

        with st.spinner(
            "🧠 Reranking relevant research content..."
        ):

            rerank_pairs = [
                [
                    clean_question,
                    chunks[idx]
                ]
                for idx in candidate_indices
            ]

            rerank_scores = (
                reranker.predict(
                    rerank_pairs
                )
            )


        # =================================================
        # BUILD RERANKED RESULTS
        # =================================================

        reranked_results = []

        for position, idx in enumerate(
            candidate_indices
        ):

            rerank_score = float(
                rerank_scores[position]
            )

            hybrid_score = float(
                hybrid_scores[idx]
            )

            faiss_score = float(
                faiss_scores.get(
                    idx,
                    0.0
                )
            )

            bm25_score = float(
                bm25_normalized[idx]
            )

            reranked_results.append(
                {
                    "idx": idx,
                    "hybrid_score": hybrid_score,
                    "rerank_score": rerank_score,
                    "faiss_score": faiss_score,
                    "bm25_score": bm25_score
                }
            )


        # =================================================
        # SORT RERANKER
        # =================================================

        reranked_results.sort(
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )


        # =================================================
        # NORMALIZED RERANK SCORE
        # =================================================

        rerank_values = [
            item["rerank_score"]
            for item in reranked_results
        ]

        normalized_rerank = (
            min_max_normalize(
                rerank_values
            )
        )


        for position, item in enumerate(
            reranked_results
        ):

            rerank_component = (
                normalized_rerank[position]
            )

            hybrid_component = (
                item["hybrid_score"]
            )

            relevance = (
                0.65 * rerank_component
                +
                0.35 * hybrid_component
            )

            item["relevance_score"] = max(
                0.0,
                min(
                    1.0,
                    relevance
                )
            )


        # =================================================
        # FINAL RESULTS
        # =================================================

        final_results = []

        used_pages = set()

        for item in reranked_results:

            idx = item["idx"]

            source = pages[idx]

            page_key = (
                source["source"],
                source["page"]
            )

            if page_key in used_pages:
                continue

            final_results.append(
                item
            )

            used_pages.add(
                page_key
            )

            if len(final_results) >= TOP_K_FINAL:
                break


        # =================================================
        # RELEVANCE CHECK
        # =================================================

        is_relevant = False

        if final_results:

            best = final_results[0]

            best_hybrid = (
                best["hybrid_score"]
            )

            best_relevance = (
                best["relevance_score"]
            )

            if (
                best_hybrid >= MIN_HYBRID_SCORE
                and
                best_relevance >= MIN_RELEVANCE_SCORE
            ):

                is_relevant = True


        # =================================================
        # RELEVANCE DISPLAY
        # =================================================

        if final_results:

            relevance_percentage = (
                final_results[0][
                    "relevance_score"
                ] * 100
            )

            st.info(
                f"🎯 Retrieval Relevance: "
                f"{relevance_percentage:.1f}%"
            )


        # =================================================
        # NOT RELEVANT
        # =================================================

        if not is_relevant:

            st.warning(
                "⚠️ I could not find sufficiently "
                "relevant information in the uploaded "
                "research papers."
            )

            st.write(
                "Try asking a question specifically "
                "related to the uploaded research papers."
            )


        # =================================================
        # GENERATE AI ANSWER
        # =================================================

        else:

            source_context = build_context(
                final_results,
                pages,
                chunks,
                MAX_CONTEXT_CHARS
            )


            # ---------------------------------------------
            # OLLAMA PROMPT
            # ---------------------------------------------

            prompt = f"""
You are a scientific research-paper
question answering assistant.

Use ONLY the research content provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not guess.
4. Do not combine unrelated information.
5. Answer only what the sources support.
6. Keep the answer concise but useful.
7. Cite the file name and page number.
8. If the answer is not supported by the sources,
respond exactly:

{FALLBACK_ANSWER}

RESEARCH CONTENT:

{source_context}

USER QUESTION:

{clean_question}

ANSWER:
"""


            # ---------------------------------------------
            # CHECK OLLAMA
            # ---------------------------------------------

            with st.spinner(
                "🔌 Checking Ollama..."
            ):

                ollama_ready, ollama_message = (
                    check_ollama()
                )


            if not ollama_ready:

                st.error(
                    f"❌ {ollama_message}"
                )

            else:

                with st.spinner(
                    "🤖 Generating AI answer..."
                ):

                    answer_text, ollama_error = (
                        generate_answer(
                            prompt
                        )
                    )


                if answer_text:

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(
                        answer_text
                    )

                else:

                    st.error(
                        "❌ AI answer generation failed."
                    )

                    st.code(
                        ollama_error
                        if ollama_error
                        else "Unknown Ollama error.",
                        language="text"
                    )


        # =================================================
        # SOURCES
        # =================================================

        if final_results:

            st.divider()

            st.subheader(
                "📚 Sources"
            )

            shown_sources = set()

            for item in final_results:

                idx = item["idx"]

                source = pages[idx]

                source_key = (
                    source["source"],
                    source["page"]
                )

                if source_key in shown_sources:
                    continue

                st.write(
                    f"📄 **{source['source']}** "
                    f"— Page **{source['page']}**"
                )

                shown_sources.add(
                    source_key
                )


            # =================================================
            # RETRIEVED CONTENT
            # =================================================

            st.subheader(
                "🔎 Retrieved Research Content"
            )

            for rank, item in enumerate(
                final_results,
                start=1
            ):

                idx = item["idx"]

                source = pages[idx]

                with st.expander(
                    f"Source {rank} — "
                    f"{source['source']} — "
                    f"Page {source['page']}"
                ):

                    st.write(
                        chunks[idx]
                    )

                    st.caption(
                        f"FAISS Similarity: "
                        f"{item['faiss_score']:.4f}"
                    )

                    st.caption(
                        f"BM25 Score: "
                        f"{item['bm25_score']:.4f}"
                    )

                    st.caption(
                        f"Hybrid Score: "
                        f"{item['hybrid_score']:.4f}"
                    )

                    st.caption(
                        f"Reranker Score: "
                        f"{item['rerank_score']:.4f}"
                    )

                    st.caption(
                        f"Relevance Score: "
                        f"{item['relevance_score']:.4f}"
                    )


# =========================================================
# NO FILE MESSAGE
# =========================================================

else:

    st.info(
        "👆 Upload one or more scientific research "
        "papers to start the RAG pipeline."
    )

    st.markdown(
        """
### 🚀 Features

- 📄 Multi-PDF document processing
- ✂️ Intelligent text chunking
- 🔢 Sentence Transformer embeddings
- 🔍 FAISS semantic search
- 🔤 BM25 keyword search
- 🔀 Hybrid retrieval
- 🧠 Cross-Encoder reranking
- 🤖 Ollama LLM generation
- 📚 Citation-based answers
- 📊 Multi-paper comparison
- 🤝 Agreement detection
- ⚡ Contradiction detection
- 🔎 Research gap detection
        """
    )
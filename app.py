import streamlit as st
import pymupdf
import faiss
import requests
import re
import time
import random

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Scientific RAG Research Engine",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Scientific RAG Research Engine")

st.caption(
    "Multi-Paper Hybrid RAG with FAISS + BM25 + Cross-Encoder Reranking"
)


# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_FAISS = 10
TOP_K_HYBRID = 10
TOP_K_RERANK = 8
TOP_K_FINAL = 5

MAX_CONTEXT_CHARS = 5000
MAX_PAPER_CONTEXT_CHARS = 2500

MIN_HYBRID_SCORE = 0.18
MIN_RELEVANCE_SCORE = 0.25

FALLBACK_ANSWER = (
    "I could not find this information in the provided research papers."
)


# =========================================================
# GEMINI CONFIG
# =========================================================

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    ""
)

GEMINI_MODEL = "gemini-3.7-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
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
# GEMINI ANSWER GENERATION
# =========================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def cached_gemini_request(prompt):

    if not GEMINI_API_KEY:

        return (
            None,
            "Gemini API key is missing. "
            "Add GEMINI_API_KEY in Streamlit Secrets."
        )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {

        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],

        "generationConfig": {
            "maxOutputTokens": 500
        }
    }

    max_retries = 2

    for attempt in range(max_retries + 1):

        try:

            response = requests.post(
                GEMINI_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            # =============================================
            # RATE LIMIT / QUOTA
            # =============================================

            if response.status_code == 429:

                if attempt < max_retries:

                    wait_time = (
                        2 ** attempt
                    ) * 5

                    wait_time += random.uniform(
                        0,
                        2
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                return (
                    None,
                    "Gemini free-tier quota/rate limit "
                    "has been reached. Please wait and "
                    "try again later."
                )

            # =============================================
            # SERVER ERRORS
            # =============================================

            if response.status_code in [500, 502, 503, 504]:

                if attempt < max_retries:

                    wait_time = (
                        2 ** attempt
                    ) * 3

                    wait_time += random.uniform(
                        0,
                        1
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                return (
                    None,
                    "Gemini service is temporarily "
                    "unavailable. Please try again later."
                )

            # =============================================
            # OTHER API ERRORS
            # =============================================

            if response.status_code != 200:

                try:

                    error_data = response.json()

                    error_message = (
                        error_data
                        .get("error", {})
                        .get("message", response.text)
                    )

                except Exception:

                    error_message = response.text

                return (
                    None,
                    f"Gemini API Error "
                    f"{response.status_code}: "
                    f"{error_message}"
                )

            # =============================================
            # PARSE RESPONSE
            # =============================================

            data = response.json()

            candidates = data.get(
                "candidates",
                []
            )

            if not candidates:

                return (
                    None,
                    "Gemini returned no answer."
                )

            content = candidates[0].get(
                "content",
                {}
            )

            parts = content.get(
                "parts",
                []
            )

            answer = ""

            for part in parts:

                answer += part.get(
                    "text",
                    ""
                )

            if not answer.strip():

                return (
                    None,
                    "Gemini returned an empty answer."
                )

            return (
                answer.strip(),
                None
            )

        except requests.exceptions.Timeout:

            if attempt < max_retries:

                time.sleep(
                    2 ** attempt
                )

                continue

            return (
                None,
                "Gemini request timed out."
            )

        except requests.exceptions.ConnectionError:

            if attempt < max_retries:

                time.sleep(
                    2 ** attempt
                )

                continue

            return (
                None,
                "Could not connect to Gemini API."
            )

        except Exception as e:

            return (
                None,
                str(e)
            )

    return (
        None,
        "Gemini request failed."
    )


# =========================================================
# GEMINI WRAPPER
# =========================================================

def generate_answer(prompt):

    return cached_gemini_request(
        prompt
    )


# =========================================================
# BUILD CONTEXT
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

        used_pages.add(
            page_key
        )

        total_chars += len(content)

    return "\n".join(
        context_parts
    )


# =========================================================
# BUILD PAPER CONTEXT
# =========================================================

def build_paper_context(
    pages,
    chunks,
    max_chars=MAX_PAPER_CONTEXT_CHARS
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

    return "\n".join(
        context_parts
    )


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
    # READ PDFs
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
                        "(no readable text)"
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
        f"✅ {processed_files} PDF(s) processed successfully!"
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

        embedding_model = (
            load_embedding_model()
        )

        reranker = (
            load_reranker()
        )


    # =====================================================
    # EMBEDDINGS
    # =====================================================

    with st.spinner(
        "🔢 Creating document embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=False,
            normalize_embeddings=True
        )

    embeddings = embeddings.astype(
        "float32"
    )


    # =====================================================
    # FAISS
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
        f"### 🔍 Vectors Stored: {index.ntotal}"
    )


    # =====================================================
    # BM25
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
            "Compare research papers for agreements, "
            "contradictions and research gaps."
        )

        comparison_type = st.selectbox(
            "Choose comparison type:",
            [
                "🤝 Agreements",
                "⚡ Contradictions",
                "🔎 Research Gaps"
            ]
        )

        compare_button = st.button(
            "🔬 Analyze Papers"
        )

        if compare_button:

            paper_context = build_paper_context(
                pages,
                chunks
            )

            if comparison_type == "🤝 Agreements":

                comparison_instruction = """
Identify the main points where the papers agree.

For each agreement:
- Explain the shared finding.
- Mention paper names.
- Mention page numbers when supported.
"""

            elif comparison_type == "⚡ Contradictions":

                comparison_instruction = """
Identify meaningful contradictions or differences.

For each difference:
- Explain Paper A.
- Explain Paper B.
- Clearly state the difference.
- Mention paper names and pages.
- Do not treat different methods automatically as contradictions.
"""

            else:

                comparison_instruction = """
Identify research gaps based ONLY on the papers.

Look for:
- Missing research areas.
- Limitations.
- Unanswered questions.
- Future research directions.

Do not invent unsupported gaps.
"""

            comparison_prompt = f"""
You are a scientific research comparison assistant.

Use ONLY the research papers provided.

RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not guess.
4. Do not fabricate citations.
5. Compare only the provided papers.
6. Clearly identify paper names.
7. Keep the answer structured.
8. Keep the answer concise.

TASK:

{comparison_instruction}

RESEARCH PAPERS:

{paper_context}

ANALYSIS:
"""

            with st.spinner(
                "🧠 Comparing research papers..."
            ):

                answer, error = (
                    generate_answer(
                        comparison_prompt
                    )
                )

            if answer:

                st.subheader(
                    "📊 Comparison Result"
                )

                st.write(
                    answer
                )

            else:

                st.error(
                    f"❌ {error}"
                )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    question = st.text_input(
        "Enter your question about the research papers:"
    )


    # =====================================================
    # SEARCH
    # =====================================================

    if question.strip():

        clean_question = question.strip()

        if len(clean_question) < 3:

            st.warning(
                "⚠️ Please enter a meaningful question."
            )

            st.stop()


        # =================================================
        # QUERY EMBEDDING
        # =================================================

        with st.spinner(
            "🔍 Running hybrid search..."
        ):

            question_embedding = (
                embedding_model.encode(
                    [clean_question],
                    normalize_embeddings=True
                )
                .astype("float32")
            )


            # =============================================
            # FAISS SEARCH
            # =============================================

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


            faiss_scores = {}

            for rank, idx in enumerate(
                faiss_indices[0]
            ):

                idx = int(idx)

                if idx < 0:
                    continue

                faiss_scores[idx] = float(
                    faiss_raw[0][rank]
                )


            # =============================================
            # BM25
            # =============================================

            tokenized_question = tokenize(
                clean_question
            )

            bm25_scores_raw = (
                bm25.get_scores(
                    tokenized_question
                )
            )

            bm25_normalized = (
                min_max_normalize(
                    list(bm25_scores_raw)
                )
            )


            # =============================================
            # HYBRID SCORE
            # =============================================

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

                hybrid_scores[idx] = (
                    0.60 * semantic_score
                    +
                    0.40 * keyword_score
                )


            # =============================================
            # TOP HYBRID
            # =============================================

            sorted_hybrid = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            candidate_indices = [
                item[0]
                for item in sorted_hybrid[
                    :TOP_K_HYBRID
                ]
            ]

            candidate_indices = (
                candidate_indices[
                    :TOP_K_RERANK
                ]
            )


        # =================================================
        # CROSS ENCODER
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
        # RERANKED RESULTS
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


        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )


        # =================================================
        # NORMALIZE RERANK SCORE
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

            relevance = (
                0.65 *
                normalized_rerank[position]
                +
                0.35 *
                item["hybrid_score"]
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
        # RELEVANCE
        # =================================================

        is_relevant = False

        if final_results:

            best = final_results[0]

            if (
                best["hybrid_score"]
                >= MIN_HYBRID_SCORE
                and
                best["relevance_score"]
                >= MIN_RELEVANCE_SCORE
            ):

                is_relevant = True


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
        # ANSWER
        # =================================================

        if not is_relevant:

            st.warning(
                "⚠️ I could not find sufficiently "
                "relevant information in the uploaded papers."
            )

        else:

            source_context = build_context(
                final_results,
                pages,
                chunks
            )

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
8. If the answer is not supported, respond exactly:

{FALLBACK_ANSWER}

RESEARCH CONTENT:

{source_context}

USER QUESTION:

{clean_question}

ANSWER:
"""

            with st.spinner(
                "🤖 Generating AI answer..."
            ):

                answer_text, error = (
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
                    f"❌ {error}"
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
# NO FILE
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
- 🤖 Gemini AI generation
- 📚 Citation-based answers
- 📊 Multi-paper comparison
- 🤝 Agreement detection
- ⚡ Contradiction detection
- 🔎 Research gap detection
"""
    )

"""
Checkpoint 8 - Streamlit UI

Upload any document (PDF/DOCX/HTML/MD/TXT) through a real button, it gets
ingested immediately, then ask questions about it -- and only it. This is
where "upload a file" stops being a CLI argument and becomes what the
project was actually supposed to do.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
import Ingest
import Query

st.set_page_config(page_title="RAGnarok", page_icon="📚")
st.title("📚 RAGnarok")
st.caption("Upload a document. Ask questions about it, and only it.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: upload + active corpus ---
with st.sidebar:
    st.header("1. Upload a document")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, HTML, Markdown, or plain text",
        type=["pdf", "docx", "html", "htm", "md", "txt"],
    )

    if uploaded_file is not None and st.button("Ingest this document", type="primary"):
        with st.spinner(f"Reading and embedding {uploaded_file.name}..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = Ingest.ingest_path(tmp_path)
            os.unlink(tmp_path)

        if "error" in result:
            st.error(result["error"])
        else:
            Query.reload_corpus()
            st.session_state.chat_history = []  # new document -- old Q&A no longer applies
            st.success(f"Ingested {result['num_chunks']} chunks from {uploaded_file.name}")
            st.rerun()

    st.divider()
    st.header("Active corpus")
    unique_sources = sorted(set(c["source"] for c in Query.chunks))
    st.write(f"**{len(unique_sources)}** file(s) - **{len(Query.chunks)}** chunks")
    for s in unique_sources[:5]:
        st.caption(f"- {os.path.basename(s)}")
    if len(unique_sources) > 5:
        st.caption(f"...and {len(unique_sources) - 5} more")

# --- Main: chat interface ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("sources"):
            with st.expander("Sources"):
                for s in entry["sources"]:
                    st.caption(s)

question = st.chat_input("Ask a question about the uploaded document...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = Query.retrieve(question)
            prompt = Query.build_prompt(question, retrieved)
            response = Query.client.chat.completions.create(
                model=os.environ["CHAT_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            answer = response.choices[0].message.content
            sources = [Query.format_source_label(c) for c in retrieved]

        st.write(answer)

        citation_check = Query.extract_citations(answer, len(retrieved))
        if citation_check["invalid"]:
            st.warning(f"Cites nonexistent source(s): {citation_check['invalid']} -- possible fabrication.")

        with st.expander("Sources"):
            for s in sources:
                st.caption(s)

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
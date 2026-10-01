import streamlit as st

st.title("Gesture RAG Assistant")

st.write("Upload any PDF to start.")

pdf_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if pdf_file is not None:
    st.success(f"PDF uploaded: {pdf_file.name}")
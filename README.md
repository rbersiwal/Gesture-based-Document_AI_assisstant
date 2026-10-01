# 🤖 Gesture RAG Assistant

A gesture-controlled Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, ask questions about them, and interact with the system using hand gestures.

The project combines **Computer Vision, Natural Language Processing, Vector Search, and Large Language Models** into one application.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDFs
- ✂️ Split documents into smaller chunks
- 🧠 Generate text embeddings using Sentence Transformers
- 🗂️ Store and search embeddings using FAISS
- 🤖 Generate answers using Llama 3.2
- 📷 Real-time hand gesture recognition using MediaPipe
- ☝️ One finger → Search
- ✌️ Two fingers → Next
- ✊ Fist → Clear
- ✋ Open palm → Stop
- 🖥️ Interactive web interface using Streamlit

---

## 🏗️ Project Architecture

```text
                    PDF Document
                         ↓
                  Text Extraction
                         ↓
                     Chunking
                         ↓
              all-MiniLM-L6-v2
                         ↓
                    Embeddings
                         ↓
                       FAISS
                         ↓
                 Relevant Context
                         ↓
                    Llama 3.2
                         ↓
                    AI Answer

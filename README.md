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
---

## Gesture Pipeline
---

                    Webcam
                       ↓
              MediaPipe Hand
                 Landmarker
                       ↓
                Hand Landmarks
                       ↓
                Finger Counting
                       ↓
              Gesture Recognition
                       ↓
        ┌────────┬────────┬────────┬────────┐
        ↓        ↓        ↓        ↓
      SEARCH    NEXT     CLEAR     STOP
       ☝️        ✌️       ✊        ✋



🧠 Models and Technologies
1. MediaPipe Hand Landmarker

Used for detecting the user's hand and identifying hand landmarks from the webcam feed.

2. all-MiniLM-L6-v2

A Sentence Transformers model used to convert PDF text and user questions into numerical embeddings.

The generated embeddings have 384 dimensions.

3. FAISS

FAISS is used for efficient similarity search between the question embedding and document embeddings.

4. Llama 3.2

Llama 3.2 is used as the Large Language Model to generate answers using the relevant information retrieved from the PDF.

It runs locally through Ollama.



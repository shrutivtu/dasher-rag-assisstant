# 🛠️ Dasher Support RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions using a combination of text and PDF documents (like an FAQ or employee manual). It uses the LangChain framework, Hugging Face models, and FAISS for semantic search.

## 🔍 How It Works

1. **Load Documents:** Both `.txt` and `.pdf` files are loaded and combined.
2. **Split Documents:** Content is chunked for better retrieval.
3. **Vector Store:** Chunks are embedded using a Sentence Transformer and stored in FAISS.
4. **Question Answering:** A `flan-t5-base` model generates answers using the retrieved context.
5. **Interactive CLI:** Users can input queries and get grounded responses pulled from the documents.

---

## 📁 Files Used

- `dasher_support_faq.txt` – Text-based FAQ or help content.
- `employeeManual.pdf` – Company documentation or internal policy.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install langchain faiss-cpu python-dotenv transformers sentence-transformers

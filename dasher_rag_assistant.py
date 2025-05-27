import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Load API key and other env variables
load_dotenv()

# 1. Load both text and PDF documents
text_loader = TextLoader("dasher_support_faq.txt")
pdf_loader = PyPDFLoader("employeeManual.pdf")

text_docs = text_loader.load()
pdf_docs = pdf_loader.load()

# Combine the documents
all_docs = text_docs + pdf_docs

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# 3. Create vectorstore using FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 4. Set up HuggingFacePipeline for flan-t5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# 5. Setup RetrievalQA
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Ask a question
while True:
    query = input("\nAsk Dasher Support (or type 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.invoke({"query": query})
    print("\n🔍 Answer:\n", result["result"])

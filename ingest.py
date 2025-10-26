# app/ingest.py
import os, glob
from collections import Counter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "safedrop"

def load_docs():
    docs = []
    # folder -> audience
    sets = [
        ("policies",       "both"),
        ("agent_sops",     "agent"),
        ("customer_faqs",  "customer"),   # << renamed folder
    ]
    for sub, audience in sets:
        folder = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):
            if not os.path.isfile(path):
                continue
            if path.lower().endswith(".pdf"):
                loaded = PyPDFLoader(path).load()
            else:
                loaded = TextLoader(path, encoding="utf-8").load()

            for d in loaded:
                d.metadata = d.metadata or {}
                d.metadata["source"] = path
                d.metadata["audience"] = audience
            docs.extend(loaded)
    return docs

if __name__ == "__main__":
    raw = load_docs()
    print(f"Loaded {len(raw)} raw docs")
    print("Docs by audience:", Counter([d.metadata["audience"] for d in raw]))
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(raw)
    print(f"Split into {len(chunks)} chunks")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
    )
    vectordb.add_documents(chunks)   # auto-persist
    print(f"Vector store ready at: {PERSIST_DIR} (collection={COLLECTION_NAME})")

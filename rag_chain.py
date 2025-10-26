import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

def _load_prompt(persona: str) -> str:
    fname = "policy_agent.txt" if persona == "agent" else "policy_customer.txt"
    with open(os.path.join(PROMPTS_DIR, fname), "r", encoding="utf-8") as f:
        return f.read()

def build_chain(persona: str = "customer"):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name="safedrop",        # ensure same name in ingest.py
        embedding_function=emb,
    )

    # Persona-aware filtering
    if persona == "agent":
        search_kwargs = {"k": 6, "filter": {"$or": [{"audience": "agent"}, {"audience": "both"}]}}
    else:
        search_kwargs = {"k": 6, "filter": {"$or": [{"audience": "customer"}, {"audience": "both"}]}}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    # Use score-threshold retriever so low-quality chunks trigger escalation
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "score_threshold": -1.0},
    )

    system = _load_prompt(persona)

    template = (
    "You are SafeDrop AI for {persona}. Answer ONLY using the provided context.\n"
    "Do NOT make up or paraphrase unseen policies. Quote directly when possible.\n"
    "If no policy matches, reply exactly: \"Escalate: no policy match found.\"\n\n"
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"persona": persona, "system": system},
    )

    llm = OllamaLLM(
        model="llama3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,  # deterministic
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
        "qa_prompt": prompt,                    # <- use qa_prompt
        "document_variable_name": "context",    # <- tell it the docs var
    },
        return_source_documents=True,
    )
    return qa


if __name__ == "__main__":
    chain = build_chain("agent")
    out = chain({"query": "Intercom broken and customer is home—what should I do?"})
    print(out["result"])
    for d in out["source_documents"]:
        print("-", d.metadata.get("source"))

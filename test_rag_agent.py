from app.rag_chain import build_chain

chain = build_chain("agent")
query = "Can I leave a signature-required parcel?"
res = chain.invoke({"query": query})

print("=== ANSWER ===")
print(res["result"])
print("\n=== SOURCES ===")
for d in res["source_documents"]:
    print("-", d.metadata.get("source"), "::", d.page_content[:160].replace("\n", " "))

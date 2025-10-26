from app.rag_chain import build_chain
import re, os

chain = build_chain("agent")
q = "Can I leave a signature-required parcel?"
res = chain.invoke({"query": q})

print("=== ANSWER ===")
print(res["result"])

print("\n=== SOURCES ===")
joined = ""
for d in res["source_documents"]:
    src = (d.metadata or {}).get("source", "unknown")
    print("-", src)
    joined += " " + d.page_content

m = re.search(r"\"([^\"]{12,})\"", res["result"])
if m:
    quote = m.group(1)
    print("\nQuoted span:", quote)
    print("Found in retrieved text?", quote in joined)
else:
    print("\nNo quote found.")

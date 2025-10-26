import streamlit as st
from rag_chain import build_chain

st.set_page_config(page_title="SafeDrop AI", page_icon="📦")

st.title("SafeDrop AI – Policy-Aware Assistant")

persona = st.sidebar.radio(
    "Who is using the assistant?",
    options=["customer", "agent"],
    format_func=lambda x: "Customer" if x == "customer" else "Delivery Agent",
)

# rebuild chain when persona changes
if "persona" not in st.session_state or st.session_state.persona != persona:
    st.session_state.persona = persona
    st.session_state.chain = build_chain(persona)

st.caption(f"Mode: **{'Customer' if persona=='customer' else 'Delivery Agent'}**")

query = st.text_input("Ask a question")
if st.button("Ask") and query.strip():
    res = st.session_state.chain.invoke({"query": user_q})
    answer = res.get("result", "").strip()
    files = []
    for d in res.get("source_documents", []):
        src = (d.metadata or {}).get("source", "")
        if src:
            files.append(os.path.basename(src))
    files_out = " | ".join(sorted(set(files))) if files else ""

    # Show the answer, then append Sources from metadata
    st.write(answer)
    if files_out:
        st.caption(f"Sources: {files_out}")
    else:
        st.caption("Sources: (none)")


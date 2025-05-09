import streamlit as st
import tempfile
from chat_with_pdf import (
    load_and_split_pdf,
    summarize_intro,
    create_vector_store,
    extract_glossary
)
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


@st.cache_resource
def generate_summary(_llm, _docs):
    return summarize_intro(_llm, _docs)

st.set_page_config(page_title="Medical PDF Assistant", layout="wide")
st.title("🩺 PDF Medical Report Chatbot")

custom_prompt = PromptTemplate.from_template("""
You are a helpful medical assistant helping a patient understand their heart echocardiogram report.

Use the context below to answer the question in **simple, clear language**. Explain any **medical terms** if needed.

If the answer is **not** in the context, say “I don’t know based on this document.”

---

Context:
{context}

Question:
{question}

Answer:
""")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    with st.spinner("🧾 Processing PDF..."):
        docs = load_and_split_pdf(file_path)

    llm = OllamaLLM(model="mistral")

    with st.spinner("🔍 Generating summary..."):
        summary = generate_summary(llm, docs)
        glossary_text = extract_glossary(llm, " ".join([doc.page_content for doc in docs[:5]]))

    vector_store = create_vector_store(docs)

    with st.sidebar:
        st.markdown("### 🩺 Glossary (AI-Generated)")
        for line in glossary_text.split("\n"):
            if ":" in line:
                term, definition = line.split(":", 1)
                st.markdown(f"**{term.strip()}**: {definition.strip()}")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    # qa_chain = build_conversational_qa_chain(llm, vector_store)

    st.subheader("📄 Auto-Summary")
    st.markdown(summary)

    st.subheader("💬 Ask Questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.qa_chain = qa_chain

    user_input = st.text_input("Your question:")

    if user_input:
        result = st.session_state.qa_chain.invoke({"question": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result["answer"]))

    for role, msg in st.session_state.chat_history[::-1]:
        st.markdown(f"**{role}:** {msg}")


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load and split the PDF
loader = PyPDFLoader("scan_echo.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# Set up local Mistral model via Ollama
llm = OllamaLLM(model="mistral")

# Combine first few chunks for summary
intro_text = " ".join([doc.page_content for doc in docs[:3]])

# Generate summary using your local model
print("\n📝 Summary of the Report:\n")
summary_prompt = f"Summarize the following medical report in plain English:\n\n{intro_text}\n\nSummary:"
summary = llm.invoke(summary_prompt)
print(summary, "\n")

# Embed and store in vector DB
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embedding_function)

# Create a retrieval QA chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Chat loop
print("Ask a question about the PDF (type 'exit' to quit):")
while True:
    query = input(">> ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.invoke({"question": query})
    print("\n" + result["answer"] + "\n")


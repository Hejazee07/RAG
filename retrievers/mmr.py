from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
        Document(
        page_content="Python is a high-level programming language known for its simplicity and readability.",
        metadata={"source": "Python Guide", "page": 1}
    ),
    Document(
        page_content="Machine Learning is a branch of Artificial Intelligence that enables systems to learn from data.",
        metadata={"source": "AI Handbook", "page": 5}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines information retrieval with large language models to provide accurate answers.",
        metadata={"source": "RAG Tutorial", "page": 10})
]

embeddings=HuggingFaceEmbeddings()

vector_store=Chroma.from_documents(docs, embeddings)

similarity_retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("Similarity Search Results\n")

similarity_docs=similarity_retriever.invoke("RAG")

for doc in similarity_docs:
    print(doc.page_content\n)

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("MMR Results\n")
mmr_docs=mmr_retriever.invoke("RAG")

for doc in mmr_docs:
    print(doc.page_content\n)
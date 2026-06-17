from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

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
        metadata={"source": "RAG Tutorial", "page": 10}
    )
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result=vectorstore.similarity_search("Deep-Learning is a branch of Artificial Intelligence.",k=2)

for r in result:
    print(r)

retriever = vectorstore.as_retriever()

docs=retriever.invoke("Explain Deep Learning?")
for d in docs:
    print(d.page_content)
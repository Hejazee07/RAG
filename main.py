from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# IMPORTANT:
# Use the EXACT SAME embedding model as create_db.py
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model,
)

print("Documents in DB:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

# max_tokens raised so the model has room to write a full, detailed summary
# instead of getting cut off after a short burst of text.
llm = ChatMistralAI(
    model="mistral-small-2506",
    max_tokens=800,
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that explains documents clearly and thoroughly.

Answer ONLY using the provided context.

Write a detailed, well-structured summary, not a short one-liner:
- Cover all the relevant points found in the context.
- Use multiple sentences (aim for at least 4-6 sentences when the context supports it).
- Organize the answer into short paragraphs or bullet points if it has multiple parts.
- Do not omit relevant details just to be brief.

If the answer is not in the context, reply:

"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}

Please provide a complete, detailed answer based on the context above.
"""
        ),
    ]
)

print("\nRAG System Ready!")
print("Type 0 to exit.\n")

while True:
    query = input("You: ").strip()

    if query == "0":
        break

    try:
        docs = retriever.invoke(query)

        if not docs:
            print("\nAI: No relevant documents found.\n")
            continue

        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query,
            }
        )

        response = llm.invoke(final_prompt)

        print("\nAI:", response.content)
        print(f"\n({len(response.content.split())} words)")
        print("-" * 80)

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__)
        print(e)
        break
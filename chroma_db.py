from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# -------------------------------------------------
# Clean invalid unicode characters
# -------------------------------------------------
def clean_text(text: str) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Remove invalid unicode surrogates
    text = text.encode("utf-8", "ignore").decode("utf-8")

    text = "".join(
        ch for ch in text
        if not (0xD800 <= ord(ch) <= 0xDFFF)
    )

    return text.strip()


# -------------------------------------------------
# Load PDF
# -------------------------------------------------
loader = PyPDFLoader("document loaders/Fundamentals.of.Deep.Learning.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")


# -------------------------------------------------
# Split PDF
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# -------------------------------------------------
# Clean chunks
# -------------------------------------------------
clean_chunks = []

for chunk in chunks:

    text = clean_text(chunk.page_content)

    if text == "":
        continue

    chunk.page_content = text
    clean_chunks.append(chunk)

print(f"Valid chunks: {len(clean_chunks)}")


# -------------------------------------------------
# Preview
# -------------------------------------------------
print("\nFirst 3 chunks:\n")

for i in range(min(3, len(clean_chunks))):
    print("=" * 60)
    print(clean_chunks[i].page_content[:200])


# -------------------------------------------------
# Embedding model
# -------------------------------------------------
print("\nLoading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# -------------------------------------------------
# Test embedding model
# -------------------------------------------------
try:

    emb = embedding_model.embed_query("Hello World")

    print("Embedding dimension:", len(emb))

except Exception as e:

    print("Embedding model failed")
    print(e)
    raise


# -------------------------------------------------
# Validate every chunk
# -------------------------------------------------
texts = []

for doc in clean_chunks:

    if not isinstance(doc.page_content, str):
        continue

    if doc.page_content.strip() == "":
        continue

    texts.append(doc.page_content)

print(f"\nTesting {len(texts)} chunks...")


# -------------------------------------------------
# Embed every chunk individually
# -------------------------------------------------
for i, text in enumerate(texts):

    try:

        embedding_model.embed_documents([text])

    except Exception as e:

        print("\nFAILED CHUNK:", i)
        print(repr(text))
        print(e)

        # print unicode characters
        print("\nUnicode characters:")

        for ch in text:
            if ord(ch) > 127:
                print(hex(ord(ch)), repr(ch))

        raise

print("\nAll chunks embedded successfully!")


# -------------------------------------------------
# Create Chroma DB
# -------------------------------------------------
print("\nCreating Chroma database...")

vectorstore = Chroma.from_documents(
    documents=clean_chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("\nChroma database created successfully!")
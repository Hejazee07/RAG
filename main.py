from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data=PyPDFLoader("document loaders/Fundamentals.of.Deep.Learning.pdf ")
docs=data.load()

template=ChatPromptTemplate.from_messages(
    [("system","You are a system that summarizes the text"),("human","{data}")]
)

model=ChatMistralAI(model = "mistral-small-2603")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(docs)

#prompt=template.format_messages(data=docs[0].page_content)

#response=model.invoke(prompt)

#print(response.content)
print(chunks)
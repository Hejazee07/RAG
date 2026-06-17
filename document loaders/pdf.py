from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/STM32 Training Manual.pdf")

docs=data.load()

print(len(docs))
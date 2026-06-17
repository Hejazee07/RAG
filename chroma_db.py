from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")
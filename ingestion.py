import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    print(sys.executable)
    print(sys.version)
    print(os.environ.get("PINECONE_API_KEY"))
    print(os.environ.get("INDEX_NAME"))

    loader = UnstructuredLoader(
        file_path="mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=10000000,
    )
    documents = loader.load()
    print("Loaded document")

    print("Spilting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("Split document into chunks")
    print(f"Number of chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Ingesting embeddings...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
    print("Ingested embeddings into Pinecone")

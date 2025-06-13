from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from typing import List

load_dotenv()

def load_documents() -> List[Document]:
    loader = TextLoader('./data/asset-classes.txt')
    documents = loader.load()
    return documents

def split_documents():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    return splitter.split_documents(documents=documents)



def create_vector_store():
    embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions = 384)
    return Chroma(
        collection_name='asset_classes_collection',
        embedding_function=embedding,
        persist_directory="./chroma_langchain_db"
    )

def add_documents_to_vector_store(vector_store):
    documents = split_documents()

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

def get_vector_store_retriever(vector_store):
    return vector_store.as_retriever(
        search_kwargs = {'k': 3}
    )
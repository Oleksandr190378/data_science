import os
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
import tiktoken
from nltk.tokenize import sent_tokenize
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from transformers import pipeline

from action_1 import text_chunks
#load_dotenv()

# Database connection settings
user = "postgres"
password = "mysecretpassword"
host = "localhost"
port = "5432"
dbname = "postgres"

CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

user_id = 1 
df_new = pd.DataFrame({
    'user_id': [user_id] * len(text_chunks),
    'content': text_chunks
})
# Load documents from DataFrame
loader = DataFrameLoader(df_new, page_content_column='content')
docs = loader.load()
print("Number of documents:", len(docs))
print("First document metadata:", docs[0].metadata if docs else "No documents")
# Add user_id to document metadata
for doc in docs:
    doc.metadata['user_id'] = user_id 

# Create vector store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="blog_posts",
    distance_strategy=DistanceStrategy.COSINE,
    connection_string=CONNECTION_STRING,
    
)


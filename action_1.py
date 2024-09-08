import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import PyPDF2
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm  # Для відображення прогресу
from sentence_transformers import SentenceTransformer
import tiktoken
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from tenacity import retry, wait_random_exponential, stop_after_attempt



'''_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])'''

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

base_path = r"C:\Users\user\Downloads\unhealthy_food.pdf"  
text_data = read_pdf(base_path)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_by_sentences(text, max_tokens=28):

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        num_tokens = len(sentence.split())
        if current_token_count + num_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += num_tokens
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = num_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


'''@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text.replace("\n", " ")
        )
        return response.data[0].embedding
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded. Retrying in a moment... Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise'''

    
def process_text_chunks(text_chunks):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    encoding = tiktoken.get_encoding("cl100k_base")

    data = []
    for text in tqdm(text_chunks):
        num_tokens = len(encoding.encode(text))
        embeddings = model.encode(text, show_progress_bar=False)
        data.append({'content': text, 'tokens': num_tokens, 'embeddings': embeddings})

    return pd.DataFrame(data)


text_chunks = chunk_text_by_sentences(text_data)
df_new = process_text_chunks(text_chunks)
print(df_new.head(10))

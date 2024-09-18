import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GPT2Tokenizer

# Helper function: get embeddings for a text
def get_embeddings(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embedding = model.encode(text, show_progress_bar=False)
    return embedding


def combine_context_and_input(context, user_input):
    return f"User: {user_input}\nAssistant:{context}"

def get_completion_from_messages(context, user_input, model='distilgpt2', temperature=0.5, max_tokens=850):
    generator = pipeline('text-generation', model=model)
    combined_text = combine_context_and_input(context, user_input)
    response = generator(  # Assuming messages are already formatted correctly
        combined_text,  # Extract content from messages
        max_length=max_tokens,
        truncation=True,
        temperature=temperature
    )
    return response[0]['generated_text']


# Helper function: Get top 3 most similar documents from the database
def get_top3_similar_docs(query_embedding, conn):
    embedding_array = np.array(query_embedding)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT 5", (embedding_array,))
    top3_docs = cur.fetchall()
    return top3_docs

# Function to process input with retrieval of most similar documents from the database
def process_input_with_retrieval(user_input, conn):
    # Step 1: Get documents related to the user input from database
    related_docs = get_top3_similar_docs(get_embeddings(user_input), conn)

    # Step 2: Create context from related documents
    context = f"Relevant information: {related_docs[0][0]} {related_docs[1][0]} {related_docs[2][0]} {related_docs[3][0]} {related_docs[4][0]}"

    # Step 3: Generate response using the context and user input
    response = get_completion_from_messages(context, user_input)
    return response

connection_string = "dbname=postgres user=postgres password=mysecretpassword host=localhost"
conn = psycopg2.connect(connection_string)
user_input = "Як впливає цукор на наш організм?"
response = process_input_with_retrieval(user_input, conn)
print("Final response:", response)


conn.close()


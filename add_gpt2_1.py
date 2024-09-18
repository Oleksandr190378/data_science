from add_gpt2 import db
from transformers import pipeline


# Initialize distilgpt2 model
generator = pipeline('text-generation', model=' gpt2')

def get_completion_from_messages(context, user_input, max_tokens=850):
    combined_text = f"Context: {context}\nUser: {user_input}\nAssistant:"
    response = generator(
        combined_text,
        max_length=max_tokens,
    )
    return response[0]['generated_text']

# Example query and response
query = "Як впливає сіль на наш організм?"
docs = db.similarity_search(query, k=3)
context = " ".join([doc.page_content for doc in docs])

# Get completion from messages
response = get_completion_from_messages(context, query)

# Display the response
print("Відповідь:")
print(response)

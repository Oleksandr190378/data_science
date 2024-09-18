import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import re

def chunk_text(text, max_tokens=28):
    words = text.split()
    total_words = len(words)
    ideal_size = int(max_tokens // (4/3))  # 1 токен ~ 3/4 слова

    new_list = []
    start = 0

    while start < total_words:
        end = start + ideal_size

        # Якщо end перевищує загальну кількість слів, додаємо решту тексту
        if end >= total_words:
            chunk_text = ' '.join(words[start:total_words])
            new_list.append(chunk_text)
            break

        # Шукаємо останній знак пунктуації перед end
        while end > start and words[end] not in '.!?':
            end -= 1

        # Якщо не знайшли знак пунктуації, просто розділяємо за ідеальним розміром
        if end == start:
            end = start + ideal_size

        chunk_text = ' '.join(words[start:end + 1])
        new_list.append(chunk_text)
        start = end + 1

    return new_list
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def chunk_text_by_sentences(text, max_tokens=28):
    """
    Розділяє текст на частини за граматичними межами, намагаючись дотримуватися обмеження за кількістю токенів.

    Args:
        text: Вхідний текст.
        max_tokens: Максимальна кількість токенів в одній частині.

    Returns:
        Список рядків, де кожен рядок - це одна частина тексту.
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        # Підраховуємо кількість токенів у реченні (приблизно)
        num_tokens = len(sentence.split())
        if current_token_count + num_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += num_tokens
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = num_tokens

    # Додаємо останній чанк, якщо він не пустий
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
# Приклад використання:
text = "Це речення. Це ін ше рече нн я! А це ще одне. ре че нн я? І ще одн е. ff gg gg. jj jj ss cc. rr дав ай ще одне.. мтмтмтмт ? мом оммомо мом оом ом. омом омо мо мтмт, сисиси. вововово"
chunks = chunk_text_by_sentences(text, max_tokens=10)
print(len(chunks))
for chunk in chunks:
    print(chunk)
# Варіант 1: Використання Sentence-Transformers
def get_embeddings_sentence_transformers(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()

# Варіант 2: Використання Hugging Face Transformers
def get_embeddings_huggingface(text):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Використовуємо середнє значення останнього прихованого стану як ембединг
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()
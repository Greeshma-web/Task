import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Web Scraping: Function to scrape meaningful content from a URL
def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 '
                      'Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract meaningful content (e.g., paragraphs and headers)
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        text_content = [para.get_text().strip() for para in paragraphs if para.get_text().strip()]
        return text_content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

# 2. Embedding Generation: Function to get embeddings for a given text
def get_embedding(text, model):
    return model.encode(text, normalize_embeddings=True)

# 3. Cosine Similarity: Function to calculate the cosine similarity
def compute_cosine_similarity(query_embedding, chunk_embeddings):
    similarities = cosine_similarity([query_embedding], chunk_embeddings)
    return similarities[0]

# 4. Response Generation: Function to generate a response using GPT-2
def generate_response_gpt2(query, context_chunk, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
    input_text = f"Question: {query}\nContext: {context_chunk}\nAnswer:"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True)
    attention_mask = torch.ones(inputs.shape, device=inputs.device)
    attention_mask[inputs == tokenizer.pad_token_id] = 0  # Adjust attention mask for padding
    
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Full Workflow
def main(url, user_query):
    # Step 1: Scrape website content
    print("Scraping website content...")
    chunks = scrape_website(url)
    if not chunks:
        return "Error: Could not scrape content from the website."
    
    # Step 2: Load Sentence-Transformer model
    print("Loading Sentence-Transformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Step 3: Generate embeddings for content chunks
    print("Generating embeddings for content chunks...")
    chunk_embeddings = [get_embedding(chunk, sentence_model) for chunk in chunks]
    
    # Step 4: Generate embedding for the user query
    print("Generating embedding for user query...")
    query_embedding = get_embedding(user_query, sentence_model)
    
    # Step 5: Compute cosine similarity
    print("Computing cosine similarities...")
    similarities = compute_cosine_similarity(query_embedding, chunk_embeddings)
    
    # Step 6: Get the most relevant content chunk
    most_similar_chunk_idx = similarities.argmax()
    most_similar_chunk = chunks[most_similar_chunk_idx]
    print(f"Most relevant content chunk:\n{most_similar_chunk}")
    
    # Step 7: Load GPT-2 model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Step 8: Generate response using GPT-2
    print("Generating response using GPT-2...")
    response = generate_response_gpt2(user_query, most_similar_chunk, gpt2_model, tokenizer)
    return response

# Example usage
if __name__ == "__main__":
    url = "https://www.uchicago.edu/"  # Example URL
    user_query = "What is the history of the University of Chicago?"
    
    # Run the full pipeline
    response = main(url, user_query)
    print("\nGenerated Response:\n", response)

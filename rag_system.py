import re
from google import genai  # Only external library for LLM
import os

def chunk_text(text, chunk_size=600, overlap=100):
    """Split text with sliding window"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def find_relevant_chunks(query, chunks, top_k=3):
    """Simple keyword-based relevance scoring"""
    query_words = set(re.findall(r'\w+', query.lower()))
    scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        # Count matching words (basic similarity)
        score = len(query_words & chunk_words)
        scores.append((score, i))
    
    # Return indices of top chunks
    return [i for _, i in sorted(scores, reverse=True)[:top_k]]

def main():
    # 1. Manual document input (no wikipedia API)
    document = """"
    Once upon a time, in a whimsical land called Veggieville, there lived a curious rabbit named Bunny. Bunny was no ordinary rabbit—she had a knack for finding strange and magical objects. One sunny morning, while hopping through the Enchanted Forest, she stumbled upon a peculiar hat lying under a giant carrot-shaped tree. The hat was no ordinary hat—it was the Magic Hat of 2000 Lines, a legendary artifact said to grant its wearer the power to weave stories, spells, and songs with just a thought.
    Bunny picked up the hat and examined it closely. It was a tall, floppy hat with shimmering silver threads that seemed to dance in the sunlight. As she placed it on her head, a voice boomed from nowhere and everywhere at once.
    "Ah, a new wearer! Welcome, Bunny. I am the Hat, and I am bound to serve you. But beware—my magic is not infinite. I can only create 2000 lines of magic before my power fades. Use them wisely!"
    Bunny's ears perked up. "2000 lines? That's a lot! What can I do with them?"
    The Hat chuckled. "Anything you can imagine! Poems, riddles, spells, even entire stories. But remember, once the lines are used up, my magic is gone forever."
    Excited, Bunny decided to test the Hat's powers. She thought of her best friend, Carrot, a cheerful orange vegetable with a knack for getting into trouble. "Hat, can you tell me a story about Carrot?"
    The Hat glowed, and a story began to unfold:
    Once, in the heart of Veggieville, there lived a carrot named Carrot who loved to explore. One day, he stumbled upon a salty cave guarded by a grumpy old grain named Salt. Salt was the keeper of the Cave of Crystals, a place filled with shimmering treasures. But Salt was lonely and bitter, and he refused to let anyone enter.
    Carrot, being the curious soul he was, decided to befriend Salt. He brought him a basket of fresh vegetables and sang him a song so sweet that even Salt's hard exterior began to melt. Slowly, Salt opened up and shared his treasures with Carrot, and the two became the unlikeliest of friends.
    Bunny clapped her paws. "That was amazing! But... how many lines did that use?"
    The Hat sighed. "That was 15 lines. You have 1985 left."
    Bunny gasped. "Oh no! I need to be more careful. I don’t want to waste your magic."
    """
    
    # 2. Simple text cleaning
    clean_text = re.sub('\s+', ' ', document).strip()
    
    # 3. Create chunks
    chunks = chunk_text(clean_text)
    
    # 4. LLM setup
    client = genai.Client(api_key=(os.getenv('GEMINI_API_KEY')))  # In practice, use environment var
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
        
        # 5. Basic retrieval using keyword overlap
        indices = find_relevant_chunks(query, chunks)
        print("\nRelevant chunks:")
        for i in indices:
            print(f"Chunk {i}: {chunks[i]}")
        context = " ".join([chunks[i] for i in indices])
        
        # 6. Simple prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Context: {context}\nQuestion: {query}\nAnswer:"
        )
        
        print(f"\nAnswer: {response.text}")

if __name__ == "__main__":
    main()
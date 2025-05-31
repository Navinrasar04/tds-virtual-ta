import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

def clean_text(text):
    return " ".join(text.strip().split())

def normalize(v):
    return v / np.linalg.norm(v)

# === Load your data ===
filename = "discourse_posts.json"  # Change to your JSON file path
with open(filename, "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# === Group posts by topic_id ===
topics = {}
for post in posts_data:
    topic_id = post["topic_id"]
    if topic_id not in topics:
        topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
    topics[topic_id]["posts"].append(post)

# Sort posts by post_number within each topic
for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

print(f"Loaded {len(posts_data)} posts across {len(topics)} topics.")

# === Initialize embedding model ===
model = SentenceTransformer("GritLM/GritLM-8x7B")

# === Function to build reply tree and extract subthreads ===
def build_reply_map(posts):
    """
    Builds a map: parent_post_number -> list of child posts
    """
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    """
    Recursively collect all posts in a subthread rooted at root_post_number
    """
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

# === Prepare embeddings for subthreads ===
embedding_data = []
embeddings = []

print("Building subthread embeddings...")

for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]
    
    reply_map, posts_by_number = build_reply_map(posts)
    
    # root posts have parent = None
    root_posts = reply_map[None]
    
    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)
        
        # Combine texts of all posts in subthread
        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(
            clean_text(p["content"]) for p in subthread_posts
        )
        
        # Embed the combined text
        emb = model.encode(combined_text, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        
        embedding_data.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "root_post_number": root_num,
            "post_numbers": [p["post_number"] for p in subthread_posts],
            "combined_text": combined_text,
        })
        embeddings.append(emb)

# Convert embeddings to numpy array for FAISS
embeddings = np.vstack(embeddings).astype("float32")

# Build FAISS index (cosine similarity with normalized vectors using inner product)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"Indexed {len(embedding_data)} subthreads.")

# === Retrieval function ===
def retrieve(query, top_k=5):
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.astype("float32")
    
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        window = embedding_data[idx]
        results.append({
            "score": float(score),
            "topic_id": window["topic_id"],
            "topic_title": window["topic_title"],
            "root_post_number": window["root_post_number"],
            "post_numbers": window["post_numbers"],
            "combined_text": window["combined_text"],
        })
    return results

query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"

results = retrieve(query, top_k=3)

print("\nTop retrieved subthreads:")
for i, res in enumerate(results, 1):
    print(f"\n[{i}] Score: {res['score']:.4f}")
    print(f"Topic ID: {res['topic_id']}, Root Post #: {res['root_post_number']}")
    print(f"Topic Title: {res['topic_title']}")
    print(f"Posts in subthread: {res['post_numbers']}")
    print("Content snippet:")
    print(res["combined_text"][:700], "...\n")  # print first 700 chars

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gen_model_name = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

def generate_answer(query, retrieved_texts, max_length=256):
    """
    Generate an answer using HF model given the query and retrieved context.
    retrieved_texts: List of strings (combined_text from retrieved results)
    """
    context = "\n\n".join(retrieved_texts)
    prompt = f"Answer the question based on the following forum discussion excerpts:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Use the top-k retrieved subthreads to generate an answer
retrieved_texts = [res["combined_text"] for res in results]
answer = generate_answer(query, retrieved_texts)

print("\nGenerated Answer:\n", answer)

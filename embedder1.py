import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Initialize local model for hybrid search
local_model = SentenceTransformer("GritLM/GritLM-8x7B")

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# Initialize ChromaDB client for vector storage
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="hybrid_discourse_embeddings",
    metadata={"description": "Hybrid embeddings using OpenAI + Local model"}
)

# === Load and process data ===
with open("discourse_posts.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

# Group posts by topic
topics = {}
for post in posts_data:
    topic_id = post["topic_id"]
    if topic_id not in topics:
        topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
    topics[topic_id]["posts"].append(post)

# Process and embed subthreads
print("Processing and embedding subthreads...")
for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]
    
    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]
    
    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)
        
        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(
            p["content"].strip() for p in subthread_posts
        )
        
        # Get OpenAI embeddings
        openai_response = client.embeddings.create(
            input=combined_text,
            model="text-embedding-3-small"
        )
        openai_embedding = openai_response.data[0].embedding
        
        # Get local model embeddings
        local_embedding = local_model.encode(combined_text, convert_to_numpy=True)
        local_embedding = local_embedding / np.linalg.norm(local_embedding)
        
        # Store in ChromaDB with both embeddings
        collection.add(
            documents=[combined_text],
            embeddings=[openai_embedding.tolist()],  # Primary embedding
            metadatas=[{
                "topic_id": topic_id,
                "topic_title": topic_title,
                "root_post_number": root_num,
                "post_numbers": [p["post_number"] for p in subthread_posts],
                "local_embedding": local_embedding.tolist()  # Store local embedding in metadata
            }],
            ids=[f"{topic_id}_{root_num}"]
        )

def hybrid_search(query, top_k=5, weight_openai=0.7):
    """
    Perform hybrid search using both OpenAI and local embeddings
    """
    # Get OpenAI query embedding
    query_openai = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_openai_emb = query_openai.data[0].embedding
    
    # Get local query embedding
    query_local_emb = local_model.encode(query, convert_to_numpy=True)
    query_local_emb = query_local_emb / np.linalg.norm(query_local_emb)
    
    # Search using OpenAI embeddings (primary)
    results = collection.query(
        query_embeddings=[query_openai_emb],
        n_results=top_k * 2  # Get more results for reranking
    )
    
    # Rerank using weighted combination of both similarities
    combined_scores = []
    for i in range(len(results["documents"])):
        doc_id = results["ids"][0][i]
        openai_score = results["distances"][0][i]  # ChromaDB returns distances
        
        # Get local embedding from metadata
        local_emb = np.array(results["metadatas"][0][i]["local_embedding"])
        local_score = np.dot(query_local_emb, local_emb)
        
        # Combine scores with weights
        combined_score = weight_openai * openai_score + (1 - weight_openai) * local_score
        combined_scores.append((combined_score, i))
    
    # Sort by combined score and get top-k
    combined_scores.sort(reverse=True)
    top_indices = [idx for _, idx in combined_scores[:top_k]]
    
    return [{
        "score": combined_scores[i][0],
        "topic_id": results["metadatas"][0][idx]["topic_id"],
        "topic_title": results["metadatas"][0][idx]["topic_title"],
        "root_post_number": results["metadatas"][0][idx]["root_post_number"],
        "post_numbers": results["metadatas"][0][idx]["post_numbers"],
        "combined_text": results["documents"][0][idx]
    } for i, idx in enumerate(top_indices)]

# Example usage
query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
results = hybrid_search(query, top_k=3)

print("\nTop hybrid search results:")
for i, res in enumerate(results, 1):
    print(f"\n[{i}] Combined Score: {res['score']:.4f}")
    print(f"Topic ID: {res['topic_id']}, Root Post #: {res['root_post_number']}")
    print(f"Topic Title: {res['topic_title']}")
    print(f"Posts in subthread: {res['post_numbers']}")
    print("Content snippet:")
    print(res["combined_text"][:700], "...\n")

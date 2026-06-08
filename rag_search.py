

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("=" * 50)
print("Task 3 - Semantic Search with Vector Embeddings")
print("=" * 50)


# Step 1 - Load the match data
df = pd.read_csv("match_data.csv")
print("\nLoaded", len(df), "matches from match_data.csv")


# Step 2 - Convert each match row into a readable English sentence
# The more descriptive the sentence, the better the search results
def make_sentence(row):
    winner = row["result"]
    # Figure out who lost
    if winner == row["team1"]:
        loser = row["team2"]
    else:
        loser = row["team1"]

    sentence = (
        row["team1"] + " vs " + row["team2"] +
        " at " + row["venue"] +
        " on " + row["date"] + ". " +
        winner + " won the match against " + loser + ". " +
        "Top scorer was " + str(row["top_scorer"]) +
        " with " + str(row["top_score"]) + " runs."
    )
    return sentence


df["sentence"] = df.apply(make_sentence, axis=1)
sentences = df["sentence"].tolist()

print("\nMatch sentences created:")
for i, s in enumerate(sentences):
    print(" ", i + 1, ".", s)


# Step 3 - Load the sentence-transformers model and create embeddings
# An embedding is just a list of numbers that represents the meaning of a sentence
print("\nLoading AI model to create embeddings...")
print("(First time may take a minute to download the model)")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(sentences)

# Convert to float32 format that FAISS needs
embeddings = embeddings.astype("float32")

# Normalize so we can use cosine similarity
faiss.normalize_L2(embeddings)

print("Embeddings created! Each sentence is now a vector of", embeddings.shape[1], "numbers")


# Step 4 - Store all embeddings in a FAISS index
# FAISS is like a super fast search engine for vectors
print("\nBuilding FAISS search index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity)
index.add(embeddings)

print("FAISS index ready with", index.ntotal, "vectors stored")


# Step 5 - Search function
def search(query, top_k=3):
    print("\nQuery:", query)
    print("-" * 45)

    # Convert the query to a vector using the same model
    query_vector = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    # Search the FAISS index for the most similar matches
    scores, indices = index.search(query_vector, top_k)

    for rank in range(top_k):
        idx = indices[0][rank]
        score = scores[0][rank]

        if idx == -1:
            continue

        similarity = round(float(score) * 100, 1)
        print("  Result", rank + 1, "- Similarity:", similarity, "%")
        print("  ", sentences[idx])
        print()


# Step 6 - Run some demo searches
print("\n" + "=" * 50)
print("Demo Searches")
print("=" * 50)

search("matches where the away team won")
search("high scoring match with a century")
search("match played in Singapore")
search("low scoring match")


# Step 7 - Let the user type their own search
print("=" * 50)
print("Try your own search (type quit to exit)")
print("=" * 50)

while True:
    user_query = input("\nEnter your search: ").strip()
    if user_query.lower() in ["quit", "exit", "q", ""]:
        print("Exiting. Goodbye!")
        break
    search(user_query)

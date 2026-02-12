# build_chroma.py

import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------------- PATH CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned", "amazon_reviews_cleaned.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_amazon_reviews")

# ---------------- MODEL CONFIG ----------------
EMBED_MODEL = "gte-large"
OLLAMA_URL = "http://127.0.0.1:11434"
BATCH_SIZE = 2000  # Larger batch = faster

# ---------------- LOAD DATA (Memory Efficient) ----------------
print("📥 Loading first 10k rows...")
df = pd.read_csv(
    CLEAN_PATH,
    nrows=10000,
    usecols=["rating", "review_text", "product", "brand", "price", "votes"]
)

print(f"📊 Rows loaded: {len(df)}")

# ---------------- PREPARE TEXTS + METADATA ----------------
texts = []
metadatas = []

for idx, row in df.iterrows():
    texts.append(f"Rating: {row['rating']}\nReview: {row['review_text']}")
    metadatas.append({
        "review_id": f"rev_{idx:08d}",
        "product": row["product"],
        "brand": row["brand"],
        "price": row["price"],
        "votes": row["votes"],
    })

# ---------------- EMBEDDING SETUP ----------------
embedding = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)

# ---------------- CREATE CHROMA ----------------
if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding,
    collection_name="amazon_reviews"
)

# ---------------- FAST BATCH INSERT ----------------
print("🚀 Embedding and storing...")

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]

    vectorstore.add_texts(
        texts=batch_texts,
        metadatas=batch_meta
    )

print("💾 Total embeddings stored:", vectorstore._collection.count())
print("🎉 Done within optimized runtime!")

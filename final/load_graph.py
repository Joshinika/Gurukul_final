# load_graph.py
import pandas as pd
import os
from neo4j import GraphDatabase

# ---------------- PATH CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "amazon_reviews_cleaned.csv"
)

# ---------------- NEO4J CONFIG ----------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ---------------- BATCH CYPHER ----------------
def create_batch(tx, batch):
    tx.run("""
        UNWIND $rows AS row
        MERGE (b:Brand {name: row.brand})
        MERGE (p:Product {name: row.product})
        SET p.price = row.price
        MERGE (b)-[:MAKES]->(p)
        CREATE (r:Review {
            review_id: row.review_id,
            rating: row.rating,
            votes: row.votes
        })
        MERGE (p)-[:HAS_REVIEW]->(r)
    """, rows=batch)

# ---------------- LOAD DATA ----------------
def load_graph():
    print("📥 Loading data in chunks...")

    batch_size = 1000
    total_inserted = 0
    max_rows = 10000  # Only first 10k rows

    with driver.session() as session:
        # Read CSV in chunks (memory efficient)
        for chunk in pd.read_csv(CLEAN_PATH, chunksize=batch_size):
            
            if total_inserted >= max_rows:
                break

            remaining = max_rows - total_inserted
            chunk = chunk.head(remaining)

            batch = []
            for idx, row in chunk.iterrows():
                batch.append({
                    "brand": row["brand"],
                    "product": row["product"],
                    "price": row.get("price"),
                    "review_id": f"rev_{total_inserted + len(batch):08d}",
                    "rating": row["rating"],
                    "votes": row["votes"]
                })

            session.execute_write(create_batch, batch)

            total_inserted += len(batch)
            print(f"Inserted: {total_inserted}")

    print("✅ Graph loaded successfully")

if __name__ == "__main__":
    load_graph()

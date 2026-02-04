import sqlite3
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)
DB = os.path.join(os.path.dirname(__file__), "leaf_info.db")

conn = sqlite3.connect(DB)
c = conn.cursor()

# Leaf Info table
c.execute("""
CREATE TABLE IF NOT EXISTS LeafInfo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    scientific_name TEXT,
    benefits TEXT,
    precautions TEXT
)
""")

# Sample records (edit as needed)
samples = [
    ("Neem", "Azadirachta indica", "Antibacterial, skin healing, anti-inflammatory", "Avoid ingestion in large amounts."),
    ("Tulsi", "Ocimum tenuiflorum", "Immunity booster, respiratory support", "Use moderate amounts in pregnancy."),
    ("AloeVera", "Aloe vera", "Skin hydration, wound healing", "External use recommended.")
]

for s in samples:
    try:
        c.execute("INSERT OR IGNORE INTO LeafInfo (name, scientific_name, benefits, precautions) VALUES (?, ?, ?, ?)", s)
    except Exception as e:
        print("Insert error:", e)

conn.commit()
conn.close()
print("Database initialized:", DB)

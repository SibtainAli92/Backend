"""Test Qdrant connection and permissions."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

print("="*60)
print("TESTING QDRANT CONNECTION")
print("="*60)

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"\n[CONFIG] Qdrant URL: {qdrant_url}")
print(f"[CONFIG] API Key present: {bool(qdrant_api_key)}")
print(f"[CONFIG] API Key length: {len(qdrant_api_key) if qdrant_api_key else 0}")

try:
    print("\n[TEST] Creating Qdrant client...")
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)

    print("[SUCCESS] Client created")

    print("\n[TEST] Getting collections list...")
    collections = client.get_collections()
    print(f"[SUCCESS] Collections found: {len(collections.collections)}")

    for col in collections.collections:
        print(f"  - {col.name}")

    print("\n[TEST] Checking if 'book_chunks' collection exists...")
    book_chunks_exists = any(c.name == "book_chunks" for c in collections.collections)
    print(f"[RESULT] book_chunks exists: {book_chunks_exists}")

    if book_chunks_exists:
        print("\n[TEST] Getting collection info...")
        info = client.get_collection("book_chunks")
        print(f"[INFO] Vectors count: {info.vectors_count}")
        print(f"[INFO] Points count: {info.points_count}")

        print("\n[TEST] Attempting a search...")
        # Try a simple search with a zero vector
        test_vector = [0.0] * 768
        results = client.search(
            collection_name="book_chunks",
            query_vector=test_vector,
            limit=1
        )
        print(f"[SUCCESS] Search completed, results: {len(results)}")
    else:
        print("[WARNING] book_chunks collection does not exist")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

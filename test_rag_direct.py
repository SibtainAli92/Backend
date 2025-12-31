"""Direct test of RAG tool to identify 403 error source."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("TESTING RAG TOOL DIRECTLY")
print("="*60)

# Check environment variables
print(f"\n[ENV] GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY'))}")
print(f"[ENV] QDRANT_URL: {os.getenv('QDRANT_URL')}")
print(f"[ENV] QDRANT_API_KEY present: {bool(os.getenv('QDRANT_API_KEY'))}")

# Import and test RAG tool
try:
    print("\n[TEST] Importing RAGTool...")
    from tools import RAGTool

    print("[TEST] Creating RAGTool instance...")
    rag_tool = RAGTool()

    print("[TEST] Testing rag_query method...")
    result = rag_tool.rag_query(
        query="What is inverse kinematics?",
        mode="rag",
        top_k=3
    )

    print(f"\n[RESULT] Success: {result.get('success')}")
    print(f"[RESULT] Error: {result.get('error')}")
    print(f"[RESULT] Sources count: {len(result.get('sources', []))}")

    if result.get('error'):
        print(f"\n[ERROR DETAILS] {result.get('error')}")

except Exception as e:
    print(f"\n[EXCEPTION] {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

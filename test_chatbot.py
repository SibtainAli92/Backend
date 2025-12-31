#!/usr/bin/env python
"""
Test script for chatbot API.
"""
import requests
import json
import time

BACKEND_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("=== Testing Health Endpoint ===")
    response = requests.get(f"{BACKEND_URL}/health")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"RAG Ready: {data['rag_ready']}")
    print(f"Vector Count: {data['startup_validation']['vector_count']}")
    print()

def test_chat():
    """Test chat endpoint."""
    print("=== Testing Chat Endpoint ===")

    questions = [
        "What is a humanoid robot?",
        "Explain forward kinematics",
        "What is the purpose of ROS2?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        payload = {
            "message": question,
            "session_id": "test_session_001",
            "top_k": 3
        }

        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        duration = time.time() - start_time

        print(f"Status: {response.status_code}")
        print(f"Duration: {duration:.2f}s")

        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:\n{data['response'][:300]}...")

            if data.get('sources'):
                print(f"\nSources ({len(data['sources'])}):")
                for i, source in enumerate(data['sources'], 1):
                    print(f"  {i}. {source.get('metadata', {}).get('book_id', 'N/A')}")
                    print(f"     {source.get('text', 'N/A')[:100]}...")
        else:
            print(f"Error: {response.text}")

        print()

if __name__ == "__main__":
    try:
        test_health()
        print("\n" + "=" * 60)
        test_chat()
        print("\n=== All Tests Completed ===")
    except Exception as e:
        print(f"Error: {e}")

"""
Test script for the OmniFinder AI search agent.
This script tests the agent with example queries from different user groups.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.agents.search_agent import SearchAgent

# Load environment variables
load_dotenv()

def test_agent():
    # Initialize the LLM and agent
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Please set your GROQ_API_KEY in the .env file")
        return
    
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    agent = SearchAgent(llm)
    
    print("🔍 OmniFinder AI - Test Suite")
    print("="*50)
    
    # Example 1 - Academic Query
    print("\n🧪 Test 1: Academic Query")
    print("Query: What are the latest developments in transformer-based language models?")
    print("-" * 50)
    
    try:
        result = agent.process_query("What are the latest developments in transformer-based language models?")
        classification = result["classification"]
        print(f"✅ Primary Tool: {classification.primary_tool}")
        if classification.secondary_tools:
            print(f"✅ Secondary Tools: {', '.join(classification.secondary_tools)}")
        print(f"✅ Reasoning: {classification.reasoning}")
        print(f"✅ Synthesized Answer Length: {len(result['synthesized_answer'])} characters")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2 - Real-time News Query
    print("\n🧪 Test 2: Real-time News Query")
    print("Query: What happened with the latest AI regulation announcements?")
    print("-" * 50)
    
    try:
        result = agent.process_query("What happened with the latest AI regulation announcements?")
        classification = result["classification"]
        print(f"✅ Primary Tool: {classification.primary_tool}")
        if classification.secondary_tools:
            print(f"✅ Secondary Tools: {', '.join(classification.secondary_tools)}")
        print(f"✅ Reasoning: {classification.reasoning}")
        print(f"✅ Synthesized Answer Length: {len(result['synthesized_answer'])} characters")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3 - General Knowledge Query
    print("\n🧪 Test 3: General Knowledge Query")
    print("Query: What is photosynthesis and how does it work?")
    print("-" * 50)
    
    try:
        result = agent.process_query("What is photosynthesis and how does it work?")
        classification = result["classification"]
        print(f"✅ Primary Tool: {classification.primary_tool}")
        if classification.secondary_tools:
            print(f"✅ Secondary Tools: {', '.join(classification.secondary_tools)}")
        print(f"✅ Reasoning: {classification.reasoning}")
        print(f"✅ Synthesized Answer Length: {len(result['synthesized_answer'])} characters")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4 - Technical Documentation Query
    print("\n🧪 Test 4: Technical Documentation Query")
    print("Query: How to implement JWT authentication in a Python web application?")
    print("-" * 50)
    
    try:
        result = agent.process_query("How to implement JWT authentication in a Python web application?")
        classification = result["classification"]
        print(f"✅ Primary Tool: {classification.primary_tool}")
        if classification.secondary_tools:
            print(f"✅ Secondary Tools: {', '.join(classification.secondary_tools)}")
        print(f"✅ Reasoning: {classification.reasoning}")
        print(f"✅ Synthesized Answer Length: {len(result['synthesized_answer'])} characters")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("✅ Testing completed!")


if __name__ == "__main__":
    test_agent()
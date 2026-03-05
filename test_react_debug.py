"""
Debug script to test ReAct agent without Streamlit
"""
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from src.agents.search_agent import SearchAgent
from src.config.agent_config import AgentConfig

# Load environment variables
load_dotenv()

print("="*60)
print("ReAct Agent Debug Test")
print("="*60)

# Initialize LLM
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    print("❌ OPENROUTER_API_KEY not found in .env file")
    exit(1)

print("\n✓ API key loaded")

# Use a free model for testing
llm = ChatOpenRouter(
    model="arcee-ai/trinity-large-preview:free",
    temperature=0,
    api_key=openrouter_api_key,
)

print(f"✓ LLM initialized with model: arcee-ai/trinity-large-preview:free")

# Create agent configuration
agent_config = AgentConfig(
    max_workers=4,
    use_react_for_complex=True,
    max_token_limit=3000,
    max_history_messages=10
)

print("✓ Agent config created")

# Create the search agent
agent = SearchAgent(llm, config=agent_config)
print("✓ Search agent created\n")

# Test query
test_query = "What is quantum computing?"
print(f"Testing with query: {test_query}")
print("="*60 + "\n")

try:
    result = agent.process_query(test_query)
    
    print("\n" + "="*60)
    print("✓ Query processed successfully!")
    print("="*60)
    print(f"\nSynthesized Answer:\n{result['synthesized_answer'][:500]}...")
    
    if "react_steps" in result:
        print(f"\n\nReAct Steps: {len(result['react_steps'])} steps")
        for i, step in enumerate(result['react_steps'][:3], 1):
            print(f"\nStep {i}: {step['action']}")
            if step['action'] == 'tool_call':
                print(f"  Tool: {step['tool']}")
                print(f"  Args: {step['arguments']}")
            elif step['action'] == 'tool_result':
                print(f"  Tool: {step['tool']}")
                print(f"  Result preview: {str(step['result'])[:100]}...")
    
except Exception as e:
    print("\n" + "="*60)
    print("❌ ERROR during query processing")
    print("="*60)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()

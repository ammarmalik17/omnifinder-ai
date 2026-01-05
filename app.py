import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from src.components.query_classifier import QueryClassification
from src.tools.search_tools import get_all_tools
from src.components.result_synthesizer import ResultSynthesizer
from src.components.conversation_memory import ConversationBufferWindowMemory
from src.agents.search_agent import SearchAgent
from langchain_core.messages import HumanMessage, AIMessage
import json


# Load environment variables
load_dotenv()

# Set up the page
st.set_page_config(
    page_title="OmniFinder AI - Intelligent Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 OmniFinder AI - Intelligent Search Agent")
st.markdown("""
This search agent intelligently routes your queries to the most appropriate search tools based on the query type:
- **Wikipedia**: General knowledge queries
- **ArXiv**: Academic papers and research
- **DuckDuckGo**: Current events and real-time information
- **Web Search**: Comprehensive multi-domain results
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    # Initialize LLM and agent
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please set your GROQ_API_KEY in the .env file")
        st.stop()
    
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    st.session_state.agent = SearchAgent(llm)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    model_option = st.selectbox(
        "Select LLM Model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    max_history = st.slider("Max History Messages", 5, 20, 10)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.agent.clear_conversation()
        st.rerun()

# Get the agent from session state
agent = st.session_state.agent

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Note: The agent will handle adding messages to its internal memory
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Process the query using the integrated agent
        with st.spinner("Processing your query..."):
            try:
                result = agent.process_query(prompt)
                classification = result["classification"]
                search_results = result["search_results"]
                synthesized_answer = result["synthesized_answer"]
                
                # Show the classification process
                with st.expander("🔍 Query Classification", expanded=True):
                    st.write(f"**Primary Tool:** {classification.primary_tool}")
                    if classification.secondary_tools:
                        st.write(f"**Secondary Tools:** {', '.join(classification.secondary_tools)}")
                    st.write(f"**Reasoning:** {classification.reasoning}")
                
                # Show search results
                with st.expander("📊 Search Results", expanded=True):
                    for search_result in search_results:
                        tool_name = search_result["tool_name"]
                        content = search_result["content"]
                        st.write(f"**Results from {tool_name}:**")
                        st.text_area(f"{tool_name} results", content, height=200)
                
                # Show synthesized answer
                with st.expander("✨ Synthesized Answer", expanded=True):
                    st.write(synthesized_answer)
                
                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": synthesized_answer})
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                error_message = "I encountered an error while processing your query. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_message})

st.markdown("---")
st.markdown("*OmniFinder AI - Your intelligent search companion*")
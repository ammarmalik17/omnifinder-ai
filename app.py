import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq


from src.agents.search_agent import SearchAgent
from src.config.agent_config import AgentConfig


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

# Define default values for configuration
max_history = 10  # Default value that matches the slider's default

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
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    # Create agent configuration based on default settings
    agent_config = AgentConfig(
        max_workers=4,
        use_react_for_complex=True,
        max_token_limit=max_history * 300,  # Approximate tokens per message
        max_history_messages=max_history
    )
    st.session_state.agent = SearchAgent(llm, config=agent_config)

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
                synthesized_answer = result["synthesized_answer"]
                
                # Check if ReAct was used (indicated by presence of react_steps)
                if "react_steps" in result:
                    # Show ReAct steps for complex queries
                    with st.expander("🧠 ReAct Reasoning Process", expanded=True):
                        st.write("**Query processed with ReAct (Reasoning and Acting) pattern**")
                        st.write("This approach dynamically reasons and acts with tools to find answers.")
                        
                        steps = result.get("react_steps", [])
                        for step in steps:
                            if step["action"] == "tool_call":
                                with st.container():
                                    st.write(f"**Step {step['step']}: Used tool** {step['tool']}")
                                    st.write(f"Arguments: {step['arguments']}")
                            elif step["action"] == "tool_result":
                                with st.container():
                                    st.write(f"**Step {step['step']}: Tool result from** {step['tool']}")
                                    with st.expander("View result details", expanded=False):
                                        st.text_area(f"{step['tool']} result", step['result'], height=150)
                            elif step["action"] == "final_response":
                                with st.container():
                                    st.write(f"**Step {step['step']}: Generated final response**")
                else:
                    # Show the classification process (traditional approach)
                    classification = result["classification"]
                    search_results = result["search_results"]
                    
                    with st.expander("🔍 Query Classification", expanded=True):
                        st.write(f"**Primary Tool:** {classification.primary_tool}")
                        if classification.secondary_tools:
                            st.write(f"**Secondary Tools:** {', '.join(classification.secondary_tools)}")
                        st.write(f"**Reasoning:** {classification.reasoning}")
                    
                    # Show search results
                    with st.expander("📊 Search Results", expanded=True):
                        if search_results:
                            for search_result in search_results:
                                tool_name = search_result.get("tool_name", "Unknown Tool")
                                content = search_result.get("content", "No content returned")
                                st.write(f"**Results from {tool_name}:**")
                                st.text_area(f"{tool_name} results", content, height=200)
                        else:
                            st.write("No search results returned.")
                
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
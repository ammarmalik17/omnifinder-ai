# OmniFinder AI - Intelligent Search Agent

A production-ready search agent application that serves three distinct user groups: students conducting academic research, professionals requiring real-time information, and general users seeking factual answers. The agent intelligently routes queries to the most appropriate search tools based on query characteristics.

## Features

- **Intelligent Query Routing**: Automatically selects the best search tools based on query type (Wikipedia for general knowledge, Arxiv for academic papers, DuckDuckGo for current events, web search for comprehensive results)
- **Multi-Tool Execution**: Concurrent execution of multiple search tools for comprehensive results
- **Result Synthesis**: Combines results from multiple tools into a coherent, well-sourced answer
- **Conversation Memory**: Maintains chat history with token limit management
- **Streamlit UI**: Clean, intuitive user interface with display of intermediate results
- **Production Ready**: Built with error handling, token management, and scalability in mind

## Architecture

The system consists of several key components:

1. **Query Classifier**: Analyzes queries to determine the most appropriate search tools
2. **Search Tools**: Wikipedia, Arxiv, DuckDuckGo, and web search capabilities
3. **Result Synthesizer**: Combines multiple results into a coherent answer
4. **Conversation Memory**: Manages chat history with token limit management
5. **Streamlit Interface**: User-friendly interface for interaction

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

### Using the Agent Programmatically

```python
from langchain_groq import ChatGroq
from src.agents.search_agent import SearchAgent

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,
    groq_api_key="your_api_key"
)

agent = SearchAgent(llm)
result = agent.process_query("Your query here")
print(result["synthesized_answer"])
```

## Example Queries

The agent handles various query types:

- **Academic Queries**: "What are the latest developments in transformer-based language models?" → Prioritizes Arxiv
- **Current Events**: "What happened with the latest AI regulation announcements?" → Prioritizes DuckDuckGo
- **General Knowledge**: "What is photosynthesis and how does it work?" → Prioritizes Wikipedia
- **Technical Documentation**: "How to implement JWT authentication in a Python web application?" → Uses web search

## Components

### Query Classification
The query classifier uses structured output to determine the most appropriate search tools based on query characteristics and intent.

### Search Tools
- **Wikipedia**: For general knowledge queries
- **Arxiv**: For academic papers and research
- **DuckDuckGo**: For current events and real-time information
- **Web Search**: For comprehensive multi-domain results

### Result Synthesis
The synthesis component combines results from multiple tools into a coherent, well-structured answer with proper source attribution.

### Conversation Memory
Two memory management strategies are implemented:
- Buffer window memory with token limits
- Summary memory for longer conversations

## Technology Stack

- **LangChain**: For LLM orchestration and tool management
- **Groq API**: For fast LLM inference (supports Llama 2, Mixtral, etc.)
- **Streamlit**: For the web interface
- **Python**: Core implementation language

## Testing

Run the test suite to verify functionality:

```bash
python test_agent.py
```

## License

This project is open source and available under the MIT License.
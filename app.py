import streamlit as st
from dotenv import load_dotenv

from backend.agents.search_agent import SearchAgent
from backend.config.agent_config import AgentConfig
from backend.core.llm_gateway import (
    create_default_llm,
    create_llm_with_model,
    get_llm_gateway,
)

# Load environment variables
load_dotenv()

# Set up the page
st.set_page_config(
    page_title="OmniFinder AI - Intelligent Search Agent", page_icon="🔍", layout="wide"
)

st.title("🔍 OmniFinder AI - Intelligent Search Agent")
st.markdown("""
This search agent intelligently routes your queries to the most appropriate search tools based on the query type:
- **Wikipedia**: General knowledge queries
- **ArXiv**: Academic papers and research
- **Web Search**: Comprehensive multi-domain results
""")

# Define default values for configuration
max_history = 10  # Default value that matches the slider's default


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    # Initialize LLM and agent using the gateway
    try:
        llm = create_default_llm()
        # Create agent configuration based on default settings
        agent_config = AgentConfig(
            max_workers=4,
            use_react_for_complex=True,
            max_token_limit=max_history * 300,  # Approximate tokens per message
            max_history_messages=max_history,
        )
        st.session_state.agent = SearchAgent(llm, config=agent_config)
        # Track the current model for dynamic switching (use gateway for provider-agnostic ID)
        st.session_state.current_model = get_llm_gateway().get_default_model()
    except ValueError as e:
        st.error(f"LLM initialization error: {str(e)}")
        st.stop()

# Sidebar for configuration
# Cache the LLM Gateway as a singleton to avoid re-initialization on every rerun
@st.cache_resource
def get_cached_gateway():
    return get_llm_gateway()

with st.sidebar:
    st.header("Configuration")

    # Get cached LLM Gateway singleton
    gateway = get_cached_gateway()

    # Benchmark/refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🤖 LLM Model")
    with col2:
        if st.button("🔄", help="Re-benchmark models", use_container_width=True):
            st.session_state.benchmark_force = True

    # Run benchmark only on first load or when refresh button is clicked
    force = st.session_state.pop("benchmark_force", False)
    if force or "benchmarked_models" not in st.session_state:
        with st.spinner("Fetching Models..."):
            st.session_state.benchmarked_models = gateway.benchmark_models(force=force)
    benchmarked = st.session_state.get("benchmarked_models", [])

    if benchmarked:
        # Build display labels with latency info
        model_labels = {
            m: f"{m}  ⚡{l:.1f}s" for m, l in benchmarked
        }
        default_index = 0
        no_models_available = False
    elif getattr(gateway, '_last_benchmark_was_fresh', False):
        # Fresh benchmark ran but returned nothing — likely daily rate limit hit
        model_labels = {}
        default_index = None
        no_models_available = True
        st.warning("Daily quota may be exhausted, or no models returned. The system is still smart, just financially grounded.")
    else:
        # Fallback: show all available models (unbenchmarked) if benchmark returned nothing
        fallback_models = gateway.get_available_models()
        if fallback_models:
            model_labels = {m: m for m in fallback_models}
            default_index = 0
            no_models_available = False
            st.caption("⚠️ Benchmark returned no results. Showing all available models unfiltered.")
        else:
            model_labels = {}
            default_index = None
            no_models_available = True
            st.warning("No models available. Check your API keys (OPENROUTER_API_KEY and/or GROQ_API_KEY).")

    model_ids = list(model_labels.keys())
    if not no_models_available:
        model_option = st.selectbox(
            "Select LLM Model",
            options=model_ids,
            format_func=lambda m: model_labels.get(m, m),
            index=default_index if model_ids else None,
        )

        # Check if model selection has changed and recreate agent if needed
        if model_option and st.session_state.current_model != model_option:
            try:
                # Create new LLM with the selected model
                new_llm = create_llm_with_model(model_option)

                # Create new agent configuration
                agent_config = AgentConfig(
                    max_workers=4,
                    use_react_for_complex=True,
                    max_token_limit=max_history * 300,
                    max_history_messages=max_history,
                )

                # Create new agent with the new LLM
                st.session_state.agent = SearchAgent(new_llm, config=agent_config)
                st.session_state.current_model = model_option

                # Show success message
                st.toast(f"Switched to {model_option}", icon="✅")

            except Exception as e:
                st.error(f"Error switching model: {str(e)}")

    if not no_models_available:
        max_history = st.slider("Max History Messages", 5, 20, 10)

        st.divider()

        # Tool and capability selection using pills
        st.subheader("🔧 Tools & Capabilities")
        tool_options = ["📖 Wikipedia", "📄 ArXiv", "🌐 Search", "🧠 ReAct"]
        selected_pills = st.pills(
            "Select tools to enable",
            options=tool_options,
            selection_mode="multi",
            default=tool_options,
            help="Choose which tools to use for search queries",
        )

        # Map selected pills to internal names
        tool_name_map = {
            "📖 Wikipedia": "wikipedia",
            "📄 ArXiv": "arxiv",
            "🌐 Search": "web_search",
            "🧠 ReAct": "react",
        }

        st.session_state.enabled_tools = []
        st.session_state.use_react_mode = False
        for pill in selected_pills:
            internal_name = tool_name_map[pill]
            if internal_name == "react":
                st.session_state.use_react_mode = True
            else:
                st.session_state.enabled_tools.append(internal_name)

        st.divider()

        def clear_chat():
            st.session_state.messages = []
            st.session_state.agent.clear_conversation()
            st.toast("Chat history cleared", icon="🗑️")

        st.button("Clear Chat History", on_click=clear_chat)

# Get the agent from session state
agent = st.session_state.agent

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything...", disabled=no_models_available):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Note: The agent will handle adding messages to its internal memory

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Process the query using the integrated agent
        with st.spinner("Processing your query..."):
            print("\n" + "=" * 60)
            print(f"User Query: {prompt}")
            print("=" * 60)
            print(f"Enabled tools: {st.session_state.enabled_tools}")
            print(f"ReAct mode: {st.session_state.use_react_mode}")

            try:
                # Pass configuration to the agent
                result = agent.process_query(
                    prompt,
                    enabled_tools=st.session_state.enabled_tools,
                    use_react=st.session_state.use_react_mode,
                )
                synthesized_answer = result["synthesized_answer"]

                print("\n✓ Query processed successfully")
                print(
                    f"Synthesized answer length: {len(synthesized_answer)} characters"
                )

                # Print detailed timing summary if available
                if "react_iterations" in result:
                    print(f"ReAct iterations: {result['react_iterations']}")
                if "classification" in result and result["classification"]:
                    cls = result["classification"]
                    print(f"Primary tool: {cls.primary_tool}")
                    if cls.secondary_tools:
                        print(f"Secondary tools: {', '.join(cls.secondary_tools)}")
                    print(f"Confidence: {cls.confidence:.2f}")

                # Check if ReAct was used (indicated by presence of react_steps)
                if "react_steps" in result:
                    # Show ReAct steps for complex queries
                    with st.expander("🧠 ReAct Reasoning Process", expanded=True):
                        st.write(
                            "**Query processed with ReAct (Reasoning and Acting) pattern**"
                        )
                        st.write(
                            "This approach dynamically reasons and acts with tools to find answers."
                        )

                        steps = result.get("react_steps", [])
                        for step in steps:
                            if step["action"] == "tool_call":
                                with st.container():
                                    st.write(
                                        f"**Step {step['step']}: Used tool** {step['tool']}"
                                    )
                                    st.write(f"Arguments: {step['arguments']}")
                            elif step["action"] == "tool_result":
                                with st.container():
                                    st.write(
                                        f"**Step {step['step']}: Tool result from** {step['tool']}"
                                    )
                                    with st.expander(
                                        "View result details", expanded=False
                                    ):
                                        st.text_area(
                                            f"{step['tool']} result",
                                            step["result"],
                                            height=150,
                                            key=f"react_tool_{step['step']}_{step['tool']}",
                                        )
                            elif step["action"] == "final_response":
                                with st.container():
                                    st.write(
                                        f"**Step {step['step']}: Generated final response**"
                                    )
                else:
                    # Show the classification process (traditional approach)
                    classification = result["classification"]
                    search_results = result["search_results"]

                    with st.expander("🔍 Query Classification", expanded=True):
                        st.write(f"**Primary Tool:** {classification.primary_tool}")
                        if classification.secondary_tools:
                            st.write(
                                f"**Secondary Tools:** {', '.join(classification.secondary_tools)}"
                            )
                        st.write(f"**Reasoning:** {classification.reasoning}")

                    # Show search results only if there are results
                    if search_results:
                        with st.expander("📊 Search Results", expanded=True):
                            for i, search_result in enumerate(search_results):
                                tool_name = search_result.get(
                                    "tool_name", "Unknown Tool"
                                )
                                content = search_result.get(
                                    "content", "No content returned"
                                )
                                st.write(f"**Results from {tool_name}:**")
                                st.text_area(
                                    f"{tool_name} results", content, height=200,
                                    key=f"{tool_name}_{i}"
                                )

                # Stream the synthesized answer progressively using true streaming
                with st.expander("✨ Synthesized Answer", expanded=True):
                    # Create a generator that yields tokens as they arrive
                    async def stream_answer():
                        try:
                            # Use the agent's streaming capability
                            async for chunk in agent.stream_synthesized_answer(
                                prompt, result
                            ):
                                yield chunk
                        except Exception as e:
                            yield f"Error during streaming: {str(e)}"

                    # Stream the response
                    try:
                        st.write_stream(stream_answer())
                    except Exception as e:
                        st.error(f"Streaming error: {str(e)}")

                # Add assistant response to session state
                st.session_state.messages.append(
                    {"role": "assistant", "content": synthesized_answer}
                )

            except Exception as e:
                print("\n❌ ERROR processing query:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback

                print(f"\nFull traceback:\n{traceback.format_exc()}")
                st.error(f"Error processing query: {str(e)}")
                error_message = "I encountered an error while processing your query. Please try again."
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

st.markdown("---")
st.markdown("*OmniFinder AI - Your intelligent search companion*")

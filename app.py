import asyncio

import streamlit as st
from dotenv import load_dotenv

from src.agents.search_agent import SearchAgent
from src.config.agent_config import AgentConfig
from src.core.llm_gateway import (
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
        # Track the current model for dynamic switching
        st.session_state.current_model = llm.model_name
    except ValueError as e:
        st.error(f"LLM initialization error: {str(e)}")
        st.stop()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Get models directly from the LLM Gateway
    gateway = get_llm_gateway()
    available_models = gateway.get_available_models()

    model_option = st.selectbox(
        "Select LLM Model", available_models, index=0 if available_models else None
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
            st.success(f"Model switched to: {model_option}")

        except Exception as e:
            st.error(f"Error switching model: {str(e)}")

    max_history = st.slider("Max History Messages", 5, 20, 10)

    st.divider()

    # Tool and capability toggles in sidebar
    st.subheader("🔧 Tools & Capabilities")
    use_wikipedia = st.toggle(
        "📚 Wikipedia", value=True, help="Search Wikipedia for general knowledge"
    )
    use_arxiv = st.toggle(
        "📄 ArXiv", value=True, help="Search academic papers on ArXiv"
    )
    use_web_search = st.toggle(
        "🌐 Web Search",
        value=True,
        help="Search web for current events and general queries",
    )
    use_react = st.toggle(
        "🧠 ReAct Mode",
        value=True,
        help="Enable advanced reasoning for complex queries",
    )

    # Store enabled tools in session state
    st.session_state.enabled_tools = []
    if use_wikipedia:
        st.session_state.enabled_tools.append("wikipedia")
    if use_arxiv:
        st.session_state.enabled_tools.append("arxiv")
    if use_web_search:
        st.session_state.enabled_tools.append("web_search")

    st.session_state.use_react_mode = use_react

    # Visual indicator of active tools in sidebar
    if st.session_state.enabled_tools:
        active_tools_str = " • ".join(
            [t.replace("_", " ").title() for t in st.session_state.enabled_tools]
        )
        if st.session_state.use_react_mode:
            active_tools_str += " • 🧠 ReAct"
        st.caption(f"Active: {active_tools_str}")

    st.divider()

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
                            for search_result in search_results:
                                tool_name = search_result.get(
                                    "tool_name", "Unknown Tool"
                                )
                                content = search_result.get(
                                    "content", "No content returned"
                                )
                                st.write(f"**Results from {tool_name}:**")
                                st.text_area(
                                    f"{tool_name} results", content, height=200
                                )

                # Stream the synthesized answer progressively using true streaming
                with st.expander("✨ Synthesized Answer", expanded=True):
                    # Use true streaming with cancellation support
                    stop_button = st.button(
                        "⏹️ Stop Streaming", key=f"stop_{len(st.session_state.messages)}"
                    )

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

                    # Stream the response with cancellation support
                    try:
                        st.write_stream(stream_answer())
                    except asyncio.CancelledError:
                        st.info("Streaming cancelled by user")
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

import streamlit as st


def render_home_tab_games(
    matches_view,
    players,
    events_view,
    plays_view,
    ga_view,
    *,
    compact: bool,
    render_games_table,
    generate_ai_team_analysis,
    ai_user_error_message,
    render_ai_debug,
) -> None:
    # Refactor-only extraction: keep widget/layout order and session_state keys identical.

    render_games_table(matches_view, compact=compact)

    # Place AI Chat Assistant under the game schedule
    st.divider()
    st.subheader("AI Assistant")
    st.caption("Ask questions about team performance and season trends")

    # Initialize chat history in session state
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.ai_chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class='ai-chat-message ai-chat-user'>
                        <strong>You:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='ai-chat-message ai-chat-assistant'>
                        <strong>AI:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_ai_debug()

    # Chat input
    c1, c2 = st.columns([4, 1])
    with c1:
        user_input = st.text_input(
            "Ask a question about the team:",
            placeholder="e.g., 'Summarize our season performance'",
            key="ai_chat_input_games",
        )
    with c2:
        send_button = st.button("Send", type="primary")

    # Quick action buttons (known data only)
    st.markdown("**Quick Actions:**")
    q2, q3 = st.columns(2)
    with q2:
        if st.button("Season Summary", help="Get a comprehensive overview of your season"):
            user_input = (
                "Provide a comprehensive summary of our season performance including strengths, "
                "weaknesses, and key insights"
            )
            send_button = True
    with q3:
        if st.button("Performance Trends", help="Analyze trends and identify improvement areas"):
            user_input = "Analyze our performance trends and identify areas for improvement"
            send_button = True

    # Process user input
    if send_button and user_input and user_input.strip():
        # Add user message to history
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})

        # Show loading spinner
        with st.spinner("AI is analyzing..."):
            # Known-data analysis only
            ai_response = generate_ai_team_analysis(
                user_input,
                matches_view,
                players,
                events_view,
                plays_view,
                ga_view,
            )

        # Add AI response to history
        if ai_response:
            st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
        else:
            st.session_state.ai_chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        ai_user_error_message(
                            "I'm sorry, I couldn't generate a response. "
                            "Please make sure you have a Groq API key configured and try again."
                        )
                    ),
                }
            )

        # Clear input and rerun to show new message
        st.rerun()

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.ai_chat_history = []
        st.rerun()

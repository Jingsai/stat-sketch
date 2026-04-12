"""Shared Groq API helpers for Streamlit chat (visualization + inference tabs)."""

try:
    from groq import Groq
except ImportError:
    Groq = None

import streamlit as st


def get_groq_api_key() -> str | None:
    """Get Groq API key from secrets or sidebar fallback."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("groq_api_key_input") or None


def call_groq(
    user_message: str,
    context: str,
    messages: list[dict],
    *,
    system_preamble: str,
) -> str:
    """Send chat completion with a system message built from preamble + context."""
    if Groq is None:
        return "The 'groq' package is not installed. Run: pip install groq"
    api_key = get_groq_api_key()
    if not api_key:
        return "To use the chat, set GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml) or enter it in the sidebar."
    try:
        client = Groq(api_key=api_key)
        system_content = system_preamble.rstrip() + "\n\nContext:\n" + context
        api_messages = [{"role": "system", "content": system_content}]
        for m in messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": user_message})
        completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=api_messages)
        return completion.choices[0].message.content or "(No response)"
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "limit" in err:
            return "Rate limit exceeded. Please wait a moment and try again."
        return f"API error: {e}"


def render_context_chat(
    storage_key: str,
    context_text: str,
    *,
    expander_title: str,
    input_placeholder: str,
    system_preamble: str,
) -> None:
    """Generic chat UI: history in session_state[f\"{storage_key}_messages\"]."""
    chat_key = f"{storage_key}_messages"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    messages = st.session_state[chat_key]
    with st.expander(expander_title, expanded=len(messages) > 0):
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        prompt = st.chat_input(input_placeholder, key=f"{storage_key}_chat_input")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            messages.append({"role": "user", "content": prompt})
            reply = call_groq(prompt, context_text, messages[:-1], system_preamble=system_preamble)
            messages.append({"role": "assistant", "content": reply})
            st.rerun()


# System prompts for each feature area
GROQ_SYSTEM_VISUALIZATION = (
    "You are a helpful data visualization assistant. The user is viewing a plot. "
    "Use the following context (plot type, parameters, and data summary) to answer their questions. "
    "Be concise and relevant."
)

GROQ_SYSTEM_INFERENCE = (
    "You are a helpful statistics instructor. The user ran a hypothesis test in a teaching app. "
    "Use the context (test name, their parameter choices, numeric results, and confidence intervals when present) "
    "to explain p-values, confidence intervals, test statistics, and what conclusions are appropriate at common "
    "alpha levels. Be accurate, concise, and avoid overclaiming causation. If context is incomplete, say so."
)

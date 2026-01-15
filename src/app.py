import streamlit as st
from openai import OpenAI
from groq import Groq
from google import genai
from core.config import config

def run_llm(provider, model_name,messages, max_tokens=500):
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    elif provider == "Google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    if provider == "Google":
        return client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages]
        ).text
    elif provider == "Groq":
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content
    elif provider == "OpenAI":
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content
    else:
        raise ValueError(f"Invalid provider: {provider}")


with st.sidebar:
    st.title("Settings")
    provider = st.selectbox("Select Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Select Model", ["gpt-5-mini", "gpt-5-nano"])
    elif provider == "Groq":
        model_name = st.selectbox("Select Model", ["llama-3.3-70b-versatile"])
    elif provider == "Google":
        model_name = st.selectbox("Select Model", ["gemini-2.5-flash"])

    st.session_state.provider = provider
    st.session_state.model_name = model_name

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist today?."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Hello! How can I assist today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = run_llm(st.session_state.provider, st.session_state.model_name, st.session_state.messages)
        # st.write(output)
        response_data=output
        answer = response_data
        # .choices[0].message.content
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    

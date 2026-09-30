import streamlit as st

from src.rag_project1.rag import create_retriever
from src.rag_project1.agent import create_agent_app

st.set_page_config(
    page_title="LangGraph RAG Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LangGraph RAG Agent")

st.write(
    "Ask questions about your document or perform calculations."
)

@st.cache_resource
def load_agent():

    retriever = create_retriever()

    agent = create_agent_app(retriever)

    return agent

agent = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something..."
)



if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )

            answer = result["messages"][-1].content

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
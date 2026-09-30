from langchain.agents import create_agent
from langchain_groq import ChatGroq

from .tools import (
    add, subtract, divide, multiply, create_search_document_tool
)

def create_agent_app(retriever):

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2
    )

    search_document = create_search_document_tool(
        retriever
    )

    tools = [
        add,
        subtract,
        multiply,
        divide,
        search_document
    ]

    agent = create_agent(
        model=model,
        tools=tools
    )

    return agent
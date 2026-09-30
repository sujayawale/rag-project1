from langchain_core.tools import tool

@tool
def add(a: float, b: float)-> float:
    """Add two numbers."""
    return a+b

@tool
def subtract(a: float, b: float)-> float:
    """Subtract two numbers."""
    return a-b

@tool
def multiply(a: float, b: float)->float:
    """Multiply two numbers."""
    return a*b

@tool
def divide(a: float, b: float)->float:
    """Divide two numbers."""
    if(b==0):
        return ValueError("Cannot divide by zero")
    return a/b


def create_search_document_tool(retriever):
    @tool
    def search_document(query: str) -> str:
        """
        Search the company document for relevant information.
        Use this tool when the user asks about information
        contained in the uploaded company document.
        """

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the document."

        results = []

        for i, doc in enumerate(docs):
            results.append(
                f"Document chunk {i + 1}:\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(results)

    return search_document

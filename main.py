import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-5.2", temperature=0)

vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=OpenAIEmbeddings()
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

## define a prompt template for the RAG process
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:
    {context}
    Question: {question}
    Please provide detail answer:"""
)


def format_documents(documents):
    """Format retrieved documents into a string for the prompt template."""
    return "\n\n".join([doc.page_content for doc in documents])


def retrieval_chain_without_lcel(query: str):
    """Simple retrieval chain without  LCEL.
    Manually retrieve documents and formats them and generates a response

    Limitations:
    -Manual step-by-step execution.
    --No built-in support for streaming responses.
    --No asynchronous support without additional code.
    --Harder to compose wth other chains.
    --More verbose and less elegant than using a chain.
    """
    # Step 1: Retrieve relevant documents from the vector store
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a string for the prompt template
    context = format_documents(docs)

    # Step 3: Format the prompt using the context and the original query
    messages = prompt_template.format_messages(context=context, question=query)

    # step 4: Generate a response from the LLM using the formatted prompt
    response = llm.invoke(messages)

    # Step 5: Return the response from the LLM
    return response.content


if __name__ == "__main__":
    print("Components initialized successfully.")
    query = "What is the pinecone in maching learning?"
    # =================================================
    # Option 0: Raw invocation without RAG
    # =================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTAITON 0:Raw LLM Invocation without RAG")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("Answer")
    print(result_raw.content)

    # =================================================
    # Option 1: Retrieval chain without LCEL
    # =================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTAITON 1: Retrieval chain without LCEL")
    print("=" * 70)
    result_retrieval_chain = retrieval_chain_without_lcel(query)
    print("Answer")
    print(result_retrieval_chain)

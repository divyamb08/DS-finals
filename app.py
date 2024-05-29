import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Set API keys
openai_api_key_ = os.environ.get('OPENAI_API_KEY')
pinecone_api_key_ = os.environ.get('PINECONE_API_KEY')
pinecone_index = os.environ.get('PINECONE_INDEX')

# Streamlit app
st.title("PDF Query Application")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key_)
    index_name = pinecone_index

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_)
    vectorstore = PineconeVectorStore.from_documents(docs, index_name=index_name, embedding=embeddings)

    # Initialize the LLM and retriever
    llm = OpenAI(openai_api_key=openai_api_key_)
    retriever = vectorstore.as_retriever()

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # or "map_reduce", "refine", etc. based on your use case
    )

    # Query input
    query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if query:
            response = qa_chain.run(query)
            st.write("Response:", response)
        else:
            st.write("Please enter a query.")
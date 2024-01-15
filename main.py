from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import tempfile
import os

st.title("Chat With Your PDF")

data = st.file_uploader("Upload your PDF")

question = st.text_input("Insert Your Question")

if data:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(data.read())
    temp_file_path = temp_file.name
    temp_file.close()

    llm = ChatOpenAI(openai_api_key= os.getenv("OPENAI_API_KEY"))

    embeddings = OpenAIEmbeddings()

    loader = PDFPlumberLoader(temp_file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template("""Responda as seguintes perguntas baseada somente no contexto fornecido:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    st.write(response["answer"])
    os.remove(temp_file_path)
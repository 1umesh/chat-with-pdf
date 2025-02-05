import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from htmlTemplates import css, bot_template, user_template
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain


from dotenv import load_dotenv
load_dotenv(override=True)


llm = AzureChatOpenAI(
    model="gpt-3.5-turbo",
    azure_deployment=os.getenv("Engine"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("API_VERSION_TO_USE"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")

)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        st.write(f"Object Type: {type(pdf)}")
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)







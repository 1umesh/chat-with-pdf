import os
import streamlit as st
from load_data import *
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv(override=True)


llm = AzureChatOpenAI(
    model="gpt-3.5-turbo",
    azure_deployment=os.getenv("Engine"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("API_VERSION_TO_USE"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")

)

st.set_page_config(page_title="Chat eith multiple documents",
                    page_icon="books")
st.write(css,unsafe_allow_html=True)
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("chat with pdfs")
user_input=st.chat_input(" write your question here")
if user_input:
    handle_userinput(user_input)

with st.sidebar:
    st.subheader(" import your documents ")
    pdf_docs= st.file_uploader(" Upload your pdf and click process ", accept_multiple_files=True)

    if st.button("process"):
        with st.spinner("processing"):
            raw_text= get_pdf_text(pdf_docs)
            text_chunks=get_text_chunks(raw_text)
            vectorstore=get_vectorstore(text_chunks)
            st.success(" Done ")
            st.session_state.conversation=get_conversation_chain(vectorstore)




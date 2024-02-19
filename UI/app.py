import time
from dotenv import find_dotenv, load_dotenv
import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


import streamlit as st
from streamlit_chat import message
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from constants import *
from run_localGPT import load_model

load_dotenv(find_dotenv())


# Streamlit app code
st.set_page_config(
    page_title='Struti AI',
    page_icon='🔖',
    layout='wide',
)

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory


## To cache resource across multiple session
@st.cache_resource
def load_embeddings():
    return HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

## To cache resource across multiple session
@st.cache_resource
def load_db():
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    return db

## To cache resource across multiple session
@st.cache_resource
def load_llm(device_type, model_id, model_basename):
    llm = load_model(device_type=device_type, model_id=model_id, model_basename=model_basename)
    return llm
    

# identify device type
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

# Define the retreiver
# load the vectorstore
if "EMBEDDINGS" not in st.session_state:
    st.session_state.EMBEDDINGS = load_embeddings()

if "DB" not in st.session_state:
    st.session_state.DB = load_db()

if "RETRIEVER" not in st.session_state:
    RETRIEVER = st.session_state.DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    st.session_state["LLM"] = load_llm(DEVICE_TYPE, MODEL_ID, MODEL_BASENAME)


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=st.session_state.LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state["QA"] = QA


st.title("Life Guru")
with st.sidebar:
    st.image("res/guruji.jpg")
    st.title("Life Guru 🙏💬")
    st.markdown(
    """
        ## About
        An AI Guru ji who can help you 24/7 by searching
        answers for your questions from vedas in realtime
        and provide you answers accordingly.
    """
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me Anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Finding Answer to your problems.."):
            response = st.session_state["QA"](prompt)
            answer = response["result"]
            response = st.write(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
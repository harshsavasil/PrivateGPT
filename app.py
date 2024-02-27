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
from langchain.vectorstores import Chroma, Milvus
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response

from constants import *
from prompt_template_utils import get_prompt_template
from run_localGPT import load_model, retrieval_qa_pipline, get_embeddings
from ingest import start_ingesting

load_dotenv(find_dotenv())


# Streamlit app code
st.set_page_config(
    page_title='Struti AI',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
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
    return get_embeddings(DEVICE_TYPE)
    # return HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

## To cache resource across multiple session
@st.cache_resource
def load_db():
    VECTOR_DATABASE = os.getenv("VECTOR_DATABASE")
    if VECTOR_DATABASE == 'MILVUS':
        db = Milvus(
            load_embeddings(),
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            collection_name=MILVUS_COLLECTION_NAME,
        )
    elif VECTOR_DATABASE == 'CHROMA':
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=load_embeddings(),
            client_settings=CHROMA_SETTINGS,
        )
    else:
        db = None
    return db

## To cache resource across multiple session
@st.cache_resource
def load_llm(device_type, model_id, model_basename):
    llm = load_model(device_type=device_type, model_id=model_id, model_basename=model_basename)
    return llm

## To cache resource across multiple session
@st.cache_resource
def get_callback_manager():
    return CallbackManager([StreamingStdOutCallbackHandler()])

def get_qa_chain(use_history):
    prompt, memory = get_prompt_template(promptTemplate_type="llama", history=use_history)
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=load_llm(DEVICE_TYPE, MODEL_ID, MODEL_BASENAME),
            chain_type="stuff",
            retriever=load_db().as_retriever(),
            return_source_documents=True,
            callbacks=get_callback_manager(),
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory
            },
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=load_llm(DEVICE_TYPE, MODEL_ID, MODEL_BASENAME),
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=load_db().as_retriever(),
            return_source_documents=True,  # verbose=True,
            callbacks=get_callback_manager(),
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
    return qa

# identify device type
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


if "QA" not in st.session_state:
    st.session_state["QA"] = get_qa_chain(True)


with st.sidebar:
    st.image("res/coder.jpg")
    st.title("System Design Expert 💻 💬")
    st.markdown(
    """
        ## About
        A System Design Expert who can help you 24/7 by searching
        answers for your questions from popular coding books in realtime
        and provide you answers accordingly.
    """
    )
    # uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type="pdf")

    # if st.button("Submit") and uploaded_files is not None:
    #     with st.spinner(text="Uploading PDF and Generating Embeddings.."):
    #         for uploaded_file in uploaded_files:
    #             bytes_data = uploaded_file.read()
    #             if uploaded_file is not None:
    #                 with open(os.path.join("SOURCE_DOCUMENTS", uploaded_file.name), "wb") as f:
    #                     f.write(uploaded_file.getbuffer())
    #         output = start_ingesting()
    #         # time.sleep(5) 
    #     st.sidebar.success('Done!')

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
        st.write(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Finding Answer to your problems.."):
            response = st.session_state["QA"](prompt)
            answer = response["result"]
            st.write(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
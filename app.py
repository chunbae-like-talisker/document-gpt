import re
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

st.set_page_config(
    page_title="Chunbae's DocumentGPT",
    page_icon="🍔",
)

# Session
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Sidebar
with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API Key")

    file = st.file_uploader(
        "Upload a txt, pdf or docx file",
        ["pdf", "txt", "docx"],
    )

    st.markdown(
        "[🍔 Chunbae's Repo](https://github.com/chunbae-like-talisker/document-gpt)"
    )

# Intro
st.markdown(
    """
# Hello!

Welcome to Chunbae's DocumentGPT!

Just upload your file from sidebar and shoot your questions.
"""
)


# LLM
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def is_valid_openai_key(key):
    return bool(re.match(r"^sk-[\S]{10,}$", key))  # Start with 'sk-'


api_key = st.session_state.get("openai_api_key")
if api_key == "":
    st.warning("Enter OpenAI API key to get started", icon="🔥")
    st.stop()
elif not is_valid_openai_key(api_key):
    st.error("Enter valid OpenAI API key to get started", icon="❌")
    st.stop()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    openai_api_key=api_key,
)

summary_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    streaming=False,
    openai_api_key=api_key,
)


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=summary_llm,
        max_token_limit=120,
        return_messages=True,
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer
            just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm


# Functions
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Uploading file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_history():
    return st.session_state["memory"].load_memory_variables({})["history"]


# Body
if file is not None:
    retriever = embed_file(file)
    send_message(
        "Ready to go! Feel free to ask me anything about your file!",
        "ai",
        save=False,
    )
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        with st.chat_message("ai"):
            result = chain.invoke(
                {
                    "context": retriever.invoke(message),
                    "history": get_history(),
                    "question": message,
                }
            )
            st.session_state["memory"].save_context(
                {"input": message},
                {"output": result.content},
            )
else:
    st.warning("Upload your file from sidebar!", icon="🔥")

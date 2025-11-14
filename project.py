import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow CPU logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Hide transformers warnings
import warnings
warnings.filterwarnings("ignore")

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import wget

# -----------------------------
# Step 1: Download file
# -----------------------------
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt"
filename = "companyPolicies.txt"
wget.download(url, out=filename)

# -----------------------------
# Step 2: Load and split text
# -----------------------------
loader = TextLoader(filename)
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_documents(documents)

# -----------------------------
# Step 3: Embed documents and store in Chroma
# -----------------------------
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
vectorstore.persist()
print("✅ Document embedded and stored in Chroma DB")

# -----------------------------
# Step 4: Initialize Ollama LLM
# -----------------------------
llm = Ollama(model="llama3")

# -----------------------------
# Step 5: Define Chat Prompt with both context + question
# -----------------------------
system_template = """You are a helpful assistant that answers questions about company policies.
Use the provided context to answer accurately and concisely.
If the answer is not found in the context, say 'I don't know.'

Context:
{context}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# -----------------------------
# Step 6: Memory and chain
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    return_source_documents=False
)

# -----------------------------
# Step 7: Interactive chat
# -----------------------------
print("\n🧠 Ask about company policies! Type 'exit', 'quit', or 'bye' to end.\n")

while True:
    query = input("You: ").strip()

    if query.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye 👋")
        break

    response = qa_chain.invoke({"question": query})
    print("Bot:", response["answer"])

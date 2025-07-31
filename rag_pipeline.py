import os
from dotenv import load_dotenv

# LangChain core components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Embedding from huggingface
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

# LLM - Google Gemini or PaLM (new way)
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def process_and_store(filename):
    model_name = "BAAI/bge-large-en-v1.5"
    embeddings = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"} 
    )

    if os.path.exists("./chroma_db"):
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    loader = TextLoader(filename, encoding='utf-8')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

def setup_rag(vectordb):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,  
        max_output_tokens=4096  
    )
    retriever = vectordb.as_retriever(
        search_type="mmr",  
        search_kwargs={"k": 10}  
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  
        retriever=retriever
    )
    return qa

def summarize(query, filename):
    vectordb = process_and_store(filename)
    qa_chain = setup_rag(vectordb)
    result = qa_chain.invoke({"query": query}) 
    print("DEBUG result:", result)
    return result.get("result", "Không có kết quả"), result.get("source_documents", [])

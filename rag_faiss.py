import ast
import operator as op
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    data_path = Path("./data/cours_demo.txt")
    if not data_path.exists():
        raise FileNotFoundError("Le fichier de données n'existe pas à l'emplacement spécifié.")
    if not openai_api_key:
        raise ValueError("Héé mathilde hé !! la clé api n'est pas charger dans ton .env !")
    
    text = data_path.read_text(encoding="utf-8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.create_documents([text])
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key, model="gpt-5-nano")
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA qui aide à répondre aux questions en utilisant les documents fournis. Si la réponse n'est pas dans les documents, dis simplement que tu ne sais pas. A la fin affiche la source avec des extraits (court)."),
        ("human", "Utilise les documents suivants pour répondre à la question :\n{context}\nQuestion : {question}. ")
    ]) 
    
    
     
    def format_docs(docs_list):
        parts = []
        for i, d in enumerate(docs_list, start=1):
            snippet = d.page_content.strip()
            parts.append(f"[{i}] {snippet}")
        return "\n\n".join(parts)

    question = "Explique ce qu'est le RAG et donne 2 bonnes pratiques."
    retrieved = retriever.invoke(question)
    context = format_docs(retrieved)

    chain = rag_prompt | llm
    res = chain.invoke({"question": question, "context": context})
    print(res.content)
    
    
if __name__ == "__main__":
    main()
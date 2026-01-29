from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("Héé mathilde hé !! la clé api n'est pas charger dans ton .env !")

    # initialisation du model 
    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    model = init_chat_model("gpt-5-nano")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es expert en cybersécurité. Tu dois m'expliquer des concepte de hack de facon claire et précise, en {duree} minutes. Si {duree} est inférieur à 20 minutes, le plan doit contenir 3 points max, sinon 6 points"),
        ("human", "Avec une technique niveau {niveau}, explique moi comment {sujet} en {duree} minutes.")
    ])
    chain = prompt | chat 
    
    res = chain.invoke({
        "sujet": "scanner un serveur avec nmap",
        "niveau": "expert",
        "duree": 10
    })

    print(res.content)

    

if __name__ == "__main__":
    main()
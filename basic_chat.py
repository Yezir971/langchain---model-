from langchain.chat_models import init_chat_model
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



    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    messages = [
        # SystemMessage(content="Tu es expert en cybersécurité. Tu dois m'expliquer des concepte de hack de facon claire et précise."),
        HumanMessage(content="Tu es prof de python donne moi des cours sur les listes")
    ]
    response = model.invoke(messages)  # Returns AIMessage
    print(response.content)
    
    # on ajoute ici le contexte dans un tableau 
    messages.append(HumanMessage(content="Donne moi un exemple de liste avec une boucle for pour itérer sur les éléments ?"))
    
    response = model.invoke(messages)  # Returns AIMessage
    print(response.content)
    

if __name__ == "__main__":
    main()
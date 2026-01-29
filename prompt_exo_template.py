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

    # initialisation du modèle 
    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    model = init_chat_model("gpt-5-nano")
    
    conversation_messages = [
        SystemMessage(content="Tu es {niveau} en cybersécurité. Tu dois m'expliquer des concepts de hack de façon claire et précise."),
        HumanMessage(content="Bonjour ! Je étudiant en cyber donne moi un cours en cybersécurité."),
    ]
    prompt = ChatPromptTemplate.from_messages(conversation_messages)
    chain = prompt | chat
    res = chain.invoke({
        "niveau": "expert"
    })
    


    history = []
    while True:
        print('assistant: ', res.content)
        
        humanInput = input("Pose ta question (ou tape 'exit' pour quitter) : ")
        if humanInput.lower() == 'exit':
            break
        if humanInput.strip() == '':
            print("Veuillez entrer une question valide.")
            continue
        if humanInput.lower() == "reset":
            conversation_messages = [
                SystemMessage(content="Tu es {niveau} en cybersécurité. Tu dois m'expliquer des concepts de hack de façon claire et précise.")
            ]
            history = []
            print("Historique réinitialisé.")
            continue

        # Ajouter le message humain à la conversation
        conversation_messages.append(HumanMessage(content=humanInput))
        
        # Crée un prompt avec tous les messages de conversation
        prompt = ChatPromptTemplate.from_messages(conversation_messages)
        
        chain = prompt | chat
        
        res = chain.invoke({
            "niveau": "expert",
            "history": history,
            "input": humanInput
        })

        print('assistant: ', res.content)
        
        # Ajouter la réponse de l'IA à la conversation
        conversation_messages.append(AIMessage(content=res.content))
        history.append(HumanMessage(content=humanInput))
        history.append(AIMessage(content=res.content))

if __name__ == "__main__":
    main()

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




# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# def main() -> None:
#     load_dotenv()
    
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Tu es un formateur {qualite}. Tu écris en Français, struturé et concret."),
#         ("human", "créé un mini plan de cours sur {sujet} pour un niveau {niveau} en {duree} minutes."),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}")
#     ])

#     chain = prompt | llm
#     user_text: str | None = None 

#     history: list = []

#     print("Chat demarré...")

#     while True:
#         res = chain.invoke({
#             "qualite": "motivé et pédagogue",
#             "sujet": "Langchain",
#             "niveau": "débutant",
#             "duree": 30,
#             "history": history,
#             "input": user_text if user_text is not None else ""
#         })
#         print("Assistant: ", res.content)

#         user_text = input("Vous:").strip()
#         if not user_text:
#             continue

#         if user_text.lower() in {"exit", "quit", "q"}:
#             print("Fin du chat.")
#             break
        
#         if user_text.lower() == "reset":
#             history.clear()
#             print("Historique réinitialisé.")
#             continue

#         history.append(("user", user_text))
#         history.append(("assistant", res.content))

# if __name__ == "__main__":
#     main()
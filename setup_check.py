import os
from dotenv import load_dotenv
# On importe ChatOpenAI directement
from langchain_openai import ChatOpenAI

def main() -> None:
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Message validé pour Mathilde !
        raise EnvironmentError('héé mathilde hééé, la clé api !!')
    
    # Correction : On définit le modèle directement dans ChatOpenAI
    # 'gpt-5-nano' est un excellent choix pour la rapidité en 2026 !
    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.7, 
        openai_api_key=api_key
    )
    
    res = llm.invoke('Réponse OK si tu me lis.')
    print(res.content)

# IMPORTANT : Ce bloc doit être aligné tout à gauche (hors de la fonction main)
if __name__ == "__main__":
    main()
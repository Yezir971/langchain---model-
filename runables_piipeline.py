from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
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
    chat = ChatOpenAI(temperature=0.9, openai_api_key=openai_api_key)

    model = init_chat_model("gpt-5-nano")
    
    
    text = """LangChain est une bibliothèque Python qui aide à orchestrer des LLM.
On peut créer des prompts réutilisables, chaîner des étapes, appeler des outils,
et construire du RAG pour répondre sur des documents.
L'approche moderne utilise LCEL (runnables) pour composer des pipelines.
    """ 
    summarization_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Tu es un expert en résumé de texte, en ajoutant à chaque fois une touche d'humour."),
        HumanMessage(content="Résume le texte suivant de manière concise : {text}")
    ])
    keypoints_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Tu es un expert en extraction de points clés."),
        HumanMessage(content="Extrait les points clés du résumé suivant : {summary}")
    ])
    
    
    quizz_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Tu es un expert en création de quiz pédagogiques, et drole avec de l'humour beauf et noir."),
        HumanMessage(content="Crée un quiz à partir des points clés suivants : {bullets}\n Quizz(A/B/C/D) avec la réponses.")
    ])
    to_str = RunnableLambda(lambda x: x.content if hasattr(x, 'content') else x) 
     
    summarization_chain = summarization_prompt | chat | to_str
    keypoints_chain = keypoints_prompt | chat | to_str
    quizz_chain = quizz_prompt | chat | to_str
    
    summary = summarization_chain.invoke({
        "txt": text,
    })
    bullets = keypoints_chain.invoke({
        "summary": summary,
    })
    quizz = quizz_chain.invoke({
        "bullets": bullets,
    })
    print("Résumé :")
    print(summary)
    print("\nPoints clés :")
    print(bullets)
    print("\nQuiz :")
    print(quizz)

if __name__ == "__main__":
    main()
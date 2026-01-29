from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

class QuizzItem(BaseModel):
    question: str = Field(..., description="The quiz question", min_length=5)
    options: List[str] = Field(..., description="A list of answer options", max_length=4)
    answer: Literal['A', 'B', 'C', 'D'] = Field(..., description="The correct answer option")
    
class LessonPack(BaseModel):
    title: str = Field(..., description="The title of the lesson", min_length=5)
    plan : list[str] = Field(..., description="The structured plan of the lesson", min_length=3)
    quizz: List[QuizzItem] = Field(..., description="A list of quiz items related to the lesson")
    estimated_minutes: int = Field(..., description="The minutes of the lecon", ge=10, le=60)
    prerequisites: List[str] = Field(..., description="The prerequier tools", min_length=2)

def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("Héé mathilde hé !! la clé api n'est pas charger dans ton .env !")

    # initialisation du model 
    llm = ChatOpenAI(temperature=0.9, openai_api_key=openai_api_key, model_name="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Tu es un expert en éducation et en création de contenu pédagogique structuré. Tu devras produire une sortie STRICTEMENT conforme au schéma demandé."),
        HumanMessage(content="Crée un pack de leçon structuré sur le sujet suivant : {sujet} et pour un niveau : {niveau}. Le pack doit inclure un titre, un plan détaillé en plusieurs points, et un quizz avec des questions à choix multiples. Réponds au format JSON conforme au modèle LessonPack. Ne numérote pas les questions du Quizz")
    ])
    
    # on veux un llm structurer donc : 
    structured_llm = llm.with_structured_output(LessonPack)
    chain = prompt | structured_llm
    
    pack = chain.invoke({
        "sujet": "l'administrtion des systèmes Linux",
        "niveau": "débutant"
    })
    
    print("Titre de la leçon :", pack.title)
    print("Plan de la leçon :")
    
    print("time : ", pack.estimated_minutes, " minutes")
    print("pré requis :", pack.prerequisites)
        
    print("PLAN :")
    for item in pack.plan:
        print("-", item)
        
    print('Quizz : ')
    for quizz in pack.quizz:
        print("Q : ", quizz.question)
        for i, option in enumerate(quizz.options, start=ord('A')):
            print(" - ", chr(i), ":", option)

            
        print("Réponse correcte : ", quizz.answer)
    

    
if __name__ == "__main__":
    main()
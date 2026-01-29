import ast
import operator as op
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. Dictionnaire des opérateurs autorisés
ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}

def _safe_eval(expr: str) -> float:
    """Évalue de manière sécurisée une expression mathématique via AST."""
    def _eval(node):
        # Utilisation de ast.Constant pour éviter la DeprecationWarning
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            fn = ALLOWED_OPS.get(type(node.op))
            if not fn:
                raise ValueError(f"Opérateur non autorisé : {type(node.op)}")
            return fn(_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            fn = ALLOWED_OPS.get(type(node.op))
            if not fn:
                raise ValueError("Opérateur non autorisé")
            return fn(_eval(node.operand))

        raise ValueError(f"Expression non autorisée : {ast.dump(node)}")

    tree = ast.parse(expr, mode='eval')
    return float(_eval(tree.body))

@tool
def calc(expression: str) -> str:
    """
    Calcule une expression mathématique simple.
    Exemple d'entrée : "2 + 2 * (3 - 1)"
    """
    try:
        result = _safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"

@tool
def word_count(text: str) -> str:
    """Compte le nombre de mots dans un texte donné."""
    count = len([word for word in text.strip().split() if word])
    return str(count)

def invoke_with_tools(llm_with_tools, prompt, question):
    # Mapping exact avec les noms des fonctions décorées
    tools_by_name = {
        "calc": calc,
        "word_count": word_count,
    }

    messages = prompt.format_messages(question=question)
    ai_msg = llm_with_tools.invoke(messages)

    if ai_msg.tool_calls:
        tool_messages = []
        for call in ai_msg.tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            if tool_fn:
                result = tool_fn.invoke(call["args"])
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        
        # Deuxième passage avec les résultats des outils
        final_msg = llm_with_tools.invoke(messages + [ai_msg] + tool_messages)
        return final_msg.content
    
    return ai_msg.content

if __name__ == "__main__":
    load_dotenv()
    # Utilisation de gpt-4o-mini pour de meilleurs résultats en tool calling
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([calc, word_count])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant qui utilise des outils mathématiques."),
        ("human", "{question}"),
    ])
    
    res = invoke_with_tools(llm, prompt, "Combien font 25 multiplié par 4 + 10 * 45 ?")
    print("Résultat :", res)
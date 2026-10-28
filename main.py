from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executot import execute_tools

MAX_ITERATIONS = 2

def draft_node(state:MessagesState):
    """ INitialise la 1ere génération avec en paramètre le state qui contient
        uniquement le HumanMessage avec la question de l'utilisateur.
    """
    first_response =  first_responder.invoke({"messages": state["messages"]})
    return {"messages": [first_response]}

def revise_node(state:MessagesState):
    """ Modifie la réponse en fonction des informations suplémentaires fournis par les outils"""
    revised_response = revisor.invoke({"messages":state['messages']})
    return {"messages": [revised_response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", "END"]:
    """ Détermine si il faut continuer à chercher ou non
        Dans le cas présent on vérifie si il y a eu + de 2 itérations d'outils.
    """
    #On compte le nombre de réponse d'outils dans le state
    count_tool_visit = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visit > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tools", execute_tools)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise",event_loop, path_map={END:END, "execute_tools":"execute_tools"})

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

res = graph.invoke({"messages":HumanMessage(content="Write about AI-Powered SOC / autonomous soc problem deomain, List startups that do that and raised capital")})
last_message = res['messages'][-1]
if (isinstance(last_message, AIMessage) and last_message.tool_calls):
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)
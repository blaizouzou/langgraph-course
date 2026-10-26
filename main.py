from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT="reflect"
GENERATE="generate"

def generation_node(state:MessageGraph):
    return  {"messages": [generate_chain.invoke({"messages":state["messages"]})]}

def reflect_node(state:MessageGraph):
    res = reflect_chain.invoke({"messages":state['messages']})
    #On prend la réponse du LLM et on le cast dans une réponse Humaine pour faire croire au LLM qu'il sagit d'une remarque d'USER
    #Les LLM sont surtout entrainé pour êtres conversationnels donc on utilise cette technique pour avoir de meilleurs résultats
    return {"messages": HumanMessage(content=res.content)}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    """
        Crée une condition pour savoir si on passe de nouveau dans le neud de reflexion ou si on peut aller vers END 
    """
    if len(state["messages"]) > 4:
        return END
    return REFLECT

#Création du Edge entre le noeud GENERATE et (REFLECT OU END) en fonction de la fonction should_continue.
#Le path Map Permet d'indiquer les sorties possible de should_continue
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
#Après la reflexion on retourne toujours à la génération.
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# print(graph.get_graph().print_ascii())

if __name__ == '__main__':
    print('hello')
    input = HumanMessage(content="Make a great random tweet better using humor on japanese stereotype")
    response = graph.invoke(input)
    print(response)
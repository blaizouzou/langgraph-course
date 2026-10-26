from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Gennerate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, adding images, style etc"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [(
        "system",
        "You are a twitter humorist influencer assistant tasked with writting excellent twitter posts."
        "Generate the best twitter post possible for the user's request"
        'If the user provides critique, respond with a revised version of your previous attempts.'
    ),
    MessagesPlaceholder(variable_name='messages')
    ]
)

llm = init_chat_model(model='ollama:qwen3.5:4b')
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

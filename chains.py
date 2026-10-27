import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# llm = init_chat_model(model='ollama:qwen3.5:4b')
# generate_chain = generation_prompt | llm
# reflect_chain = reflection_prompt | llm

llm = ChatOpenAI(model="gpt-6-luna")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

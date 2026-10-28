from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from schemas import AnswerQuestion, ReviseAnswer
import datetime


load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are an expert researcher
        Current time: {time}
        1. {first_instruction}
        2. reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.
        """), MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(time=lambda:datetime.datetime.now().isoformat())

revise_instructions = """
            Revise your previous answer using the new information
            -You should use the previous critique to add important information to your answer?
            -You must include numerical citation in your revised answer to ensure it can be verified
            -add a "References" section to the bottom of your answer (Wich does not count towards the word limit) like so
                -[1] https://example.com
                -[2] https://example.com
            -You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words
"""


first_responder_prompt_template = actor_prompt_template.partial(first_instruction="provide a detailed answer around 250 words")

first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revisor = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == '__main__':
    print("Hello Reflexion")
    human_message = HumanMessage(content="Write about AI-Powered SOC / autonomous soc problem deomain, List startups that do that and raised capital")
    chain = (first_responder | parser_pydantic)
    res = chain.invoke(input={"messages": [human_message]})
    print(res)
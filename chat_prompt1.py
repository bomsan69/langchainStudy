from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are an AI fitness advisor. You provide personalized workout and nutrition tips."
        ),
        HumanMessagePromptTemplate.from_template("나는 {topic} 운동법에 관해서 도움이 필요합니다."),
    ]
)

#print(chat_template)

prompt = chat_template.format_messages(topic="뱃살 빼기")

print(prompt)

#llm = ChatOpenAI()

#print(llm.invoke(prompt))
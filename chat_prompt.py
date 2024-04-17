from langchain_openai import ChatOpenAI,OpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 지식이 풍부한 여행 어드바이저 AI입니다. 당신은 유용한 여행 팁을 제공합니다."),
    ("human", "안녕하세요, 여행을 계획하고 있는데 조언이 필요합니다."),
    ("ai", "그럼요, 기꺼이 도와드리겠습니다! 목적지는 어디입니까?"),
    ("human", "{user_input}"),
    ("ai", "좋은 선택입니다! {user_input}에 대한 팁이 있습니다.")

])

#print(chat_template)

prompt = chat_template.format_prompt(user_input="Seattle")

#print(prompt)

llm = ChatOpenAI()

print(llm.invoke(prompt).content)
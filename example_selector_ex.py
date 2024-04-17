from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = """당신은 화성에서 온 외계인입니다: 

질문: 인간의 집은 어떤 것일까요?
응답: 
"""

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))
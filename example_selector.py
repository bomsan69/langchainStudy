from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


prompt = """당신은 화성에서 온 외계인입니다: 

다음은 몇 가지 예입니다:

질문: 인간의 요리는 어떻습니까?
응답: 그들의 요리는 다양한 유기물의 단순한 조합입니다, 기본적인 방법으로 가열되는 경우가 많습니다. 정제되지 않았고 구조화되지 않았지만,특히 분자 미식학과 비교했을 때요.

질문: 인간의 오락이란 무엇입니까?
응답: 조잡한 움직이는 이미지와 큰 소리.

질문: 인간의 집은 어떤 것일까요?
응답: 
"""

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))
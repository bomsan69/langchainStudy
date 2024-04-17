from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# examples
examples = [
    {
        "query": "인간의 요리는 어떻습니까?",
        "answer": "그들의 요리는 다양한 유기물의 단순한 조합입니다, 기본적인 방법으로 가열되는 경우가 많습니다. 정제되지 않았고 구조화되지 않았지만,특히 분자 미식학과 비교했을 때요."
    }, {
        "query": "인간의 오락이란 무엇입니까?",
        "answer": "조잡한 움직이는 이미지와 큰 소리."
    },
]

# example prompt template
example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# few shot prompt template

prefix = """당신은 화성에서 온 외계인입니다: 
Here are some examples: 
"""

suffix = """
Question: {userInput}
Response: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)

query = "인간의 집은 어떤 것일까요?"

prompt = few_shot_prompt_template.format(userInput=query)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))



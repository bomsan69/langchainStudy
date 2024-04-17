from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv

load_dotenv()

# examples
# Let's use more examples with different lengths.

examples = [
    {
        "Constellation": "양자리",
        "fortune": "살랑살랑 훈풍이 불어 오는가 싶더니 그것이 모두 착각이었구나 싶어지는 날입니다. 계획을 세워놓았던 일들이 자꾸 뒤로 미루어지게 되고, 그러다 보니 일을 하고자 하는 의욕도 자꾸 수그러들게 됩니다."
    },
     {
        "Constellation": "물병자리",
        "fortune": "문서를 이용하여 하는 일 또는 주변 사람들의 동의가 있어야 하는 일을 진행하기에 적당한 날입니다. 가만히 머리로 생각만 할 것이 아니라 생각한 것을 실천으로 옮기기에 적당한 하루이니 지금 바로 움직이도록 하세요."
    },
    {
        "Constellation": "물고기자리",
        "fortune": "모든 것이 자기 마음대로 되는 것은 아님을 깨닫게 될 수 있는 날입니다. 어리석은 사람들은 때를 알지 못하고 함부로 날뛰다 불 속에 타 죽는 불나비와 같다고 하였습니다. 현명하게 심사숙고 후에 움직일 수 있도록 해야 합니다."
    },
    {
        "Constellation": "쌍둥이자리",
        "fortune": "세상사가 모두 자신의 마음대로만 되는 것은 아닙니다. 간혹 자신의 마음과는 전혀 상관없는 일이 벌어지게 되는 경우도 많습니다. 하지만 그렇다고 해도 쉽사리 포기해서는 안 됩니다."
    },
     {
        "Constellation": "황소자리",
        "fortune": "자신의 현재 자리에 불만이 쌓이는 날입니다. 당장 자리를 박차고 나가고 싶어지며, 현재의 자신에 대해서도 탐탁지 않아 하게 됩니다. 욕구 불만이 강하니 엉뚱한 생각이 들 수도 있습니다."
    },
]

# example prompt template
example_template = """
Birthday: {Constellation}
Fortune: {fortune}
"""

example_prompt = PromptTemplate(
    input_variables=["Birthday", "Fortune"],
    template=example_template
)

# few shot prompt template

# LengthBasedExampleSelector
MAX_LENGTH = 100

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=MAX_LENGTH
)

# few shot prompt template

prefix = """You are an AI fortune teller. It says things that make people happy. 
Here are some examples: 
"""

suffix = """
Constellation: {userInput}
Response: """

new_prompt_template  = FewShotPromptTemplate(
    example_selector=example_selector,  # examples 대신 example_selector를 사용
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)

query = "나의 별자리는 황소자리입니다."

prompt = new_prompt_template.format(userInput=query)

print(prompt)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))



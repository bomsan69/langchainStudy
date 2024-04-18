import gradio as gr
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv

load_dotenv()

examples = [
    {
        "Constellation": "양자리",
        "fortune": "오늘의 양지리(3월21일~4월19일) 운세는 살랑살랑 훈풍이 불어 오는가 싶더니 그것이 모두 착각이었구나 싶어지는 날입니다. 계획을 세워놓았던 일들이 자꾸 뒤로 미루어지게 되고, 그러다 보니 일을 하고자 하는 의욕도 자꾸 수그러들게 됩니다."
    },
     {
        "Constellation": "물병자리",
        "fortune": "오늘의 물병자리 운세는 문서를 이용하여 하는 일 또는 주변 사람들의 동의가 있어야 하는 일을 진행하기에 적당한 날입니다. 가만히 머리로 생각만 할 것이 아니라 생각한 것을 실천으로 옮기기에 적당한 하루이니 지금 바로 움직이도록 하세요."
    },
    {
        "Constellation": "물고기자리",
        "fortune": "오늘의 물고기자리((1월20일~2월18일)) 운세는 모든 것이 자기 마음대로 되는 것은 아님을 깨닫게 될 수 있는 날입니다. 어리석은 사람들은 때를 알지 못하고 함부로 날뛰다 불 속에 타 죽는 불나비와 같다고 하였습니다. 현명하게 심사숙고 후에 움직일 수 있도록 해야 합니다."
    },
    {
        "Constellation": "쌍둥이자리",
        "fortune": "오늘의 쌍둥이자리(5월 21일~6월 21일) 운세는 세상사가 모두 자신의 마음대로만 되는 것은 아닙니다. 간혹 자신의 마음과는 전혀 상관없는 일이 벌어지게 되는 경우도 많습니다. 하지만 그렇다고 해도 쉽사리 포기해서는 안 됩니다."
    },
     {
        "Constellation": "황소자리",
        "fortune": "오늘의 황소자리(4월 20일~5월 20일) 운세는 자신의 현재 자리에 불만이 쌓이는 날입니다. 당장 자리를 박차고 나가고 싶어지며, 현재의 자신에 대해서도 탐탁지 않아 하게 됩니다. 욕구 불만이 강하니 엉뚱한 생각이 들 수도 있습니다."
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

prefix = """당신은 별자리 점술가로 오늘의 운세를 고객에게 이야기 해주어야 합니다 .
사람들에게 희망과 용기를 주는 말을 해 주세요. 고객이 생일을 입력하면 별자리를 찾아서 대답하고 별자리를 입력하면 별자리 날짜도 답변에 포함 시켜 주세요.  
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


def gentAnswer(message):

    prompt = new_prompt_template.format(userInput=message)

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)

    respons = llm.invoke(prompt)

    html_res=f"<div style='height: 400px; background-color: pink; font-size: 26px'>{respons}</div>"


    return "",html_res

js="""
function changeFavicon(url) {
    // Find the existing favicon element
    var favicon = document.querySelector('link[rel="shortcut icon"]') || document.createElement('link');
    favicon.type = 'image/x-icon';
    favicon.rel = 'shortcut icon';
    favicon.href = url;

    // Replace or append the new favicon to <head>
    var head = document.querySelector('head');
    var oldFavicon = head.querySelector('link[rel="shortcut icon"]');
    if (oldFavicon) {
        head.removeChild(oldFavicon);
    }
    head.appendChild(favicon);
}

// Usage
changeFavicon('./favicon.ico');

"""
app = gr.Blocks()


with app:

    gr.Markdown("<h1><center> 오늘의 운세가 궁금한가요? </center></h1>")


    message = gr.Textbox(placeholder="당신의 별자리를 입력해 주세요.",container=False)

    #result = gr.Textbox(label="Result",container=False, lines=4, show_copy_button=True)
    #result = gr.Markdown()
    result2 = gr.HTML("""
              <div style='height: 400px; background-color: pink;'></div>
            """)
    btn = gr.Button("운세 보기")
    
    
    btn.click(gentAnswer, inputs=[message], outputs=[message,result2])


app.launch()
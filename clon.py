import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()



def start_conversation():

    #PromptTemplate(input_variables=['history', 'input'], template='The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:')¶
    
    conversation = ConversationChain(
        llm=ChatOpenAI(temperature=0),
        memory=ConversationBufferMemory(),
        verbose=False
    )

    return conversation

conversation = start_conversation()

#creat the app

app = FastAPI()

#defie the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

#index page route
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()
    
#매번 10번쨰 대화 후 오른쪽 창에 대화 요약 표시
COUNTER = 0

@app.post("/process")
async def process_data(item:dict):
    global COUNTER
    # get the input
    user_input = item.get("data","Np data received")
    print("user_input",user_input)

    # send the  input to the convseration chain
    response  = conversation.predict(input=user_input)
  
    if COUNTER == 0 or COUNTER % 10 == 0:
        # if necessary, update the summary
        summary = conversation.memory.buffer
        return {"message": response, "summary": summary}

    COUNTER += 1
    print("COUNTER:",COUNTER)
    return {"message": response}

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

load_dotenv()

chat = ChatOpenAI(temperature=0.1, model="gpt-4")

actor_name_schema = ResponseSchema(name="actor_name",
                                   description="This refers to the name of the actor involved in the film scene.")

actor_pickup_location_schema = ResponseSchema(name="actor_pickup_location",
                                       description="This is the location where the actor will be picked up or collected from before the shoot.")

scene_location_schema = ResponseSchema(name="scene_location",
                                       description="This is the actual location where the scene will be filmed.")

scene_time_schema = ResponseSchema(name="scene_time",
                                       description="This denotes the scheduled time when the scene is set to be shot.")

response_schema = [
    actor_name_schema,
    actor_pickup_location_schema,
    scene_location_schema,
    scene_time_schema
]

#print(response_schema)

star_constellation = ResponseSchema(name="star_constellation", description="Star constellation");
star_fortune = ResponseSchema(name="star_fortune", description="Constellation horoscope results");

start_schema = [
    star_constellation,
    star_fortune
]

#print(start_schema)

output_parser1 = StructuredOutputParser.from_response_schemas(response_schema)

format_instructions1 = output_parser1.get_format_instructions()

#print(format_instructions1)

output_parser2 = StructuredOutputParser.from_response_schemas(start_schema)

format_instructions2 = output_parser2.get_format_instructions()


#print(format_instructions2)

text1="""
Tomorrow promises to be an eventful day as we're scheduled to shoot a 
pivotal scene featuring the renowned actor, Brad Pitt. 
The scene requires the unique charm of Central Park's Bethesda Terrace, 
precisely at 2 PM, to capture the essence of the storyline under the 
gentle afternoon sun. To ensure a timely start, Brad will be picked up 
from The Ritz-Carlton, New York, at 12:30 PM. This will allow for a smooth 
journey, factoring in city traffic and any last-minute preparations needed 
upon arrival. Please confirm that all necessary equipment and personnel 
are on-site and ready by 1 PM, to fully utilize the natural light and 
setting of the location.
"""

text2="""
오늘의 양자리(3월21일~4월19일) 운세는 살랑살랑 훈풍이 불어 오는가 싶더니 그것이 모두 착각이었구나 싶어지는 날입니다. 계획을 세워놓았던 일들이 자꾸 뒤로 미루어지게 되고, 그러다 보니 일을 하고자 하는 의욕도 자꾸 수그러들게 됩니다
"""


template = """
From the following text message, extract the following information:

text message: {text}
{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(template)
messages = prompt_template.format_messages(text=text1, format_instructions=format_instructions1)

response = chat.invoke(messages)

print(response.content)
print(type(response.content))

messages = prompt_template.format_messages(text=text2, format_instructions=format_instructions2)

response = chat.invoke(messages)

print(response.content)

json_result = json.loads(response.content)
print(json_result)
from langchain_openai import ChatOpenAI,OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


prompt_template = PromptTemplate.from_template(
    "Modify the following {original_recipe} to suit a {dietary_preference} and adjust it to a {difficulty_level} cooking level." 
)

print(prompt_template)

prompt = prompt_template.format(
    original_recipe="This recipe details how to make a classic pork sausage. It includes pork, sugar, and black pepper.", 
    dietary_preference="Korean style Bulgogi sausage", 
    difficulty_level="Simplified for beginners"
    )


print(prompt)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))
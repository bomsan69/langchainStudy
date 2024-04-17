from langchain_openai import OpenAI,OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# examples
# Let's use more examples with different lengths.

examples = [
    {
        "query": "What is human cuisine like?",
        "answer": "Their cuisine is a simplistic combination of various organic matter, often heated in rudimentary ways. It's unrefined and unstructured, especially compared to our molecular gastronomy."
    }, {
        "query": "What is human entertainment?",
        "answer": "Crude moving images and loud sounds."
    }, {
        "query": "What do humans use for transportation?",
        "answer": "Humans rely on archaic and inefficient rolling contraptions they proudly call 'cars.' These are remarkably primitive compared to our teleportation beams and anti-gravity vessels."
    }, {
        "query": "How do humans communicate with each other?",
        "answer": "They use a very basic form of communication involving the modulation of sound waves, referred to as 'speech.' Astonishingly primitive compared to our telepathic links."
    }, {
        "query": "How do humans maintain health?",
        "answer": "Consuming organic compounds and performing physical movements."
    }, {
        "query": "What is human education?",
        "answer": "They engage in a very basic form of knowledge transfer in places called 'schools.' It's a slow and inefficient process compared to our instant knowledge assimilation."
    }, {
        "query": "How do humans manage their societies?",
        "answer": "Through chaotic and inefficient systems."
    }, {
        "query": "What is human art?",
        "answer": "Their art is a primitive expression through physical mediums like paint and stone, lacking the sophistication of our holographic emotion sculptures."
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

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to 
    # measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings 
    # and do a similarity search over.
    Chroma,
    # The number of examples to produce.
    k=1,
)

# few shot prompt template

prefix = """당신은 화성에서 온 외계인입니다: 
Here are some examples: 
"""

suffix = """
Question: {userInput}
Response: """

similar_prompt   = FewShotPromptTemplate(
    example_selector=example_selector,  # examples 대신 example_selector를 사용
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)

query = "What are human homes like?"

prompt = similar_prompt.format(userInput=query)

print(prompt)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm.invoke(prompt))



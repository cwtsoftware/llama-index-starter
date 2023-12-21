import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    set_global_service_context
)
from llama_index.llms import OpenAI

llm = OpenAI()
service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context)

# Agenti u LlamaIndex-u su pokretani llm-om, to su inteligentni chatbot-ovi sa znanjem, sposobni obavljati zadatke 
# nad danim dokumentima za čitanje ali i pisanje podataka, preciznije, imaju pristup prilagođenim alatima. Alati
# mogou biti funkcije, drugi llm-ovi, query_engine i sl.

# za ovaj primjer definiraju se fukcije (alati) množenja i zbrajanja 
from llama_index.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Pomnoži dva broja i vrati rezultat"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Zbroji dva broja i vrati rezultat"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# definira se agent i proslijeđuju mu se alati
from llama_index.agent import OpenAIAgent
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], 
    llm=llm, 
    verbose=True
)

# poziv prema agentu i izvori
response = agent.chat("What is (121 * 3) + 42?")
print(str(response))
print("------------------------------")
print(response.sources)

# # stream
# response = agent.stream_chat(
#     "What is 121 * 2? Once you have the answer, use that number to write a"
#     " story about a group of mice."
# )
# response_gen = response.response_gen
# for token in response_gen:
#     print(token, end="")

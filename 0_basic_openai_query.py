import os
from llama_index.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# llama-index pruža mogućnost direktnog upita prema openai
# za ovaj slučaj se generalno preporuća direktni api call prema openai

# opći primjer

res = OpenAI().complete("Koja je najveća luka u Hrvatskoj?")
print(res)

# opći primjer - streaming

llm = OpenAI()
resp = llm.stream_complete("Koja je najveća luka u Hrvatskoj?")
for res in resp:
    print(res)

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
    llm=llm,
    system_prompt="Respond in Croatian language"
)
set_global_service_context(service_context)

# chat_engine se koristi u slučajevima kada komunikacija sa višestrukim iteracijama tijekom kojih 
# se prati povijest razgovora u usporedbi sa query_engine koji se koirsti u jednostrukim pitanjima

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
  documents
)

# chat_engine ima više načina rada. Tijekom ovog tutoriala će se koristiti chat_mode="context" i "condense_plus_context"
# "context" pronalazi čvorove koji su najsličniji upitu
# "condense_question" pregledava povijest razgovora i ponovno ispiše korisničku poruku da bude upit za indeks.
# "condense_plus_context" je kombinacija "context" i "condense_question"

# jedonstavan chat_engine koji daje rezultate na temelju pronađenog konteksta 
chat_engine = index.as_chat_engine(chat_mode="context", similarity_top_k=4)
response = chat_engine.chat("o cemu se radi")
print(response)

# izvori
for node in response.source_nodes:
    print("\n----------------------------------------------------")
    print("node.text -> ", node.text)

# # printanje svih radnji prije nego se dobije rezultat
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # dodaje se jednostavna memorija kao buffer
# from llama_index.memory import ChatMemoryBuffer
# memory = ChatMemoryBuffer.from_defaults()

# chat_engine = index.as_chat_engine(
#   chat_mode="condense_plus_context",
#   memory=memory,
# )
# streaming_response = chat_engine.stream_chat("o cemu se radi")
# streaming_response.print_response_stream()

# # može se isčitati da se uzima u obzir povijest razgovora i pitanje se preformulira u "Koji token koristi Bitcoin?"
# streaming_response = chat_engine.stream_chat("koji token koristi") 
# streaming_response.print_response_stream()

# # dokaz o povijesti
# streaming_response = chat_engine.stream_chat("koje je bilo moje prvo pitanje?")
# streaming_response.print_response_stream()
    
# # također je moguće započeti sesiju razovora; ovo će se koristit za lakše testiranje i evaluacije
# chat_engine.chat_repl()
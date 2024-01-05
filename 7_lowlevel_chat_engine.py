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

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
    documents
)

# chat_engine generalno dodaje nove funkcionalnosti i gradi se na temelju query_engine
# kao što je rečeno u prijašnjem modulu, bitna razlika je u dodavanju povijesti tijekom cijelog razgovora
# do sada su se koristili uglavnom high-level api koncepti, za korištenje prilagođene povijesti razgovora
# mora se koristiti low-level api kompozicija 
from llama_index.llms import ChatMessage, MessageRole
from llama_index.chat_engine.condense_plus_context import (
    CondensePlusContextChatEngine,
)

# definira se povijest razgovora; prvo se definira poruka korisnika onda asistenta
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Koji je nabolji kripto projekt?",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Bitcoin."),
]

# definira se query_engine
query_engine = index.as_query_engine()

# definira se retriver
# Retriveri su odgovorni za dohvaćanje najrelevantnijeg konteksta s obzirom na korisnički upit iz indeksa
retreiver = index.as_retriever()

# definira se CondensePlusContextChatEngine
# ovo je ekvivalentno sa "chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")" ali
# sa low-level api kompozicijom; u kojem se definira svaki aspekt "ručno"
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retreiver,
    query_engine=query_engine,
    chat_history=custom_chat_history,
    verbose=True # prikazuje pitanje i kontekst
)
  
# definira se pitanje
streaming_response = chat_engine.stream_chat("mozes li mi reci vise o tome?")
streaming_response.print_response_stream()

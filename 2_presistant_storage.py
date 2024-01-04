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

llm = OpenAI(
  system_prompt="Always respond in croatian language"
)

service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context) # postavlja se globalni service_context i uvijek se koristi za indekse

# da nebi morali kreirati novi vektorski zapis svaki puta kada se pokrene skripta moguće je spremiti zapis lokalno
# te ga učitati svaki puta kada se pokrene skirpta

# ako već postoji kreirani zapis učitaj taj zapis
from llama_index import (
    StorageContext,
    load_index_from_storage,
)
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2021"
    )
    index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

print(index_loaded)

# ako ne kreiraj novi
if not index_loaded:
    # učitaj podatke
    docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    # kreiraj indeks
    index = VectorStoreIndex.from_documents(docs)

    # spremi zapis
    index.storage_context.persist(persist_dir="./storage/2021")

query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Poslovnice u inozemstvu")
streaming_response.print_response_stream()

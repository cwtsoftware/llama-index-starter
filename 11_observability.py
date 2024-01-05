import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# u ovom modulu će se prikazati na koji način je moguće pratiti upite
# llama-index nudi mnogo integracija za praćenje, ovdje se koriti Traceloop 

# inicijalizira se Traceloop sa api_key-em koji se dobiva nakon registracija na njihovoj stranici (https://app.traceloop.com/)
# za potrebe ovog tutoriala, Traceloop je besplatan
# nakon poslanog upita, svi koraci se zapisujui moguće ih je pregledati na stranicama Traceloop-a
from traceloop.sdk import Traceloop
traceloop_key = os.getenv("TRACELOOP_API_KEY")

Traceloop.init(disable_batch=True, api_key=traceloop_key)

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
set_global_service_context(service_context)

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

# ako ne kreiraj novi
if not index_loaded:
    docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    index = VectorStoreIndex.from_documents(docs)

    index.storage_context.persist(persist_dir="./storage/2021")

query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Poslovnice u inozemstvu")
streaming_response.print_response_stream()

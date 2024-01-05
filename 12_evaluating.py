import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# u ovom modulu biti će prikazano na koji se način evaluira relevantnost rezultata i dohvaćenih čvorova
# postoje komplekcniji načini eveluacije ali za potrebe ovog tutoriala će početni biti dovoljan

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

if not index_loaded:
    docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    index = VectorStoreIndex.from_documents(docs)

    index.storage_context.persist(persist_dir="./storage/2021")

# inicijaliziraj evaluator
from llama_index.evaluation import FaithfulnessEvaluator
evaluator = FaithfulnessEvaluator(service_context=service_context)

# pošalji upit
query_engine = index.as_query_engine()
response = query_engine.query("Poslovnice u inozemstvu")
print(response)

# evauliraj relevantnost odgovora na temelju rezultata
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))

# evaluiraj relevantnost dohvaćenih čvorova
response_str = response.response
for source_node in response.source_nodes:
    eval_result = evaluator.evaluate(
        response=response_str, contexts=[source_node.get_content()]
    )
    print(str(eval_result.passing))

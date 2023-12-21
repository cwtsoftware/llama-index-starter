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

# učitaj dokumente
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

# učitaj prvi indeks za godinu 2021
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2021"
    )
    twentyone_index = load_index_from_storage(storage_context)

    index_one_loaded = True
except:
    index_one_loaded = False

if not index_one_loaded:
    twentyone_docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    twentyone_index = VectorStoreIndex.from_documents(twentyone_docs)

    twentyone_index.storage_context.persist(persist_dir="./storage/2021")

# učitaj drugi indeks za godinu 2022
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2022"
    )
    twentytwo_index = load_index_from_storage(storage_context)

    index_two_loaded = True
except:
    index_two_loaded = False

if not index_two_loaded:
    twentytwo_docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2022-CA.pdf"]
    ).load_data()

    twentytwo_index = VectorStoreIndex.from_documents(twentytwo_docs)

    twentytwo_index.storage_context.persist(persist_dir="./storage/2022")

# kreiraj query_engine
twentyone_engine = twentyone_index.as_query_engine(similarity_top_k=3)
twentytwo_engine = twentytwo_index.as_query_engine(similarity_top_k=3)

# kreiraj query_engine alate koje će koristiti agent
from llama_index.tools import QueryEngineTool, ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=twentyone_engine,
        metadata=ToolMetadata(
            name="croatia_airlines_report_2021",
            description=(
                "Provides information about Croatia Airlines bussines report for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=twentytwo_engine,
        metadata=ToolMetadata(
            name="croatia_airlines_report_2022",
            description=(
                "Provides information about Croatia Airlines bussines report for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

# kreiraj agenta i proslijedi alate
from llama_index.agent import OpenAIAgent
agent = OpenAIAgent.from_tools(
    query_engine_tools, 
    verbose=True,
    system_prompt="""
        - The questions are for Croatia Airlines report for years 2021 and 2022. 
        - Remember to always use available tools
        - explain how did you get the final answer 
    """
)

# postavi pojedinačno pitanje
res = agent.query("which year had more employes")
print(res)

# # razgovaraj sa agentom
# agent.chat_repl()

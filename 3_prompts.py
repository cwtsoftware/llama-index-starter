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
from llama_index.prompts import PromptTemplate

llm = OpenAI(
  system_prompt="Always respond in croatian language"
)
service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context)

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
  documents
)

# za pronalaženje relevantnih čvorova i dobivanje točnih odgovora potrebno je 
# imati dobar upit (eng. prompt), idealna situacija jest kada se llm-u daju 
# točni koraci koje treba izvršiti i to se radi sa prilagođenim upitima

# prema default postavkama query_engine koristi "text_qa_template" za svaki upit, a
# ako je dohvaćeni kontekst predugačak za samo jedan llm poziv, onda se koristi i 
# "refine_template" 
query_engine = index.as_query_engine()
prompts_dict = query_engine.get_prompts()
for k, v in prompts_dict.items():
    print("Prompt key -> ", k)
    print(v.get_template())
    print(f"\n\n")

# # za mijenjanje šablone upita uvijek je potrebno imati varijable "context_str" za dohvaćeni kontekst i 
# # "query_str" za upit
# custom_qa_prompt = PromptTemplate(
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the query in the style of a Shakespeare play.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# query_engine = index.as_query_engine(
#     text_qa_template=custom_qa_prompt
# )
# prompts_dict = query_engine.get_prompts()
# for k, v in prompts_dict.items():
#     print("Prompt key -> ", k)
#     print(v.get_template())
#     print(f"\n\n")

# # ponekad je potrebno dinamički mijenjati upit
# qa_prompt_tmpl_str = """\
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Please write the answer in the style of {tone_name}
# Query: {query_str}
# Answer: \
# """

# prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
# partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")
# query_engine = index.as_query_engine(
#     text_qa_template=partial_prompt_tmpl
# )
# # prompts_dict = query_engine.get_prompts()
# # for k, v in prompts_dict.items():
# #     print("Prompt key -> ", k)
# #     print(v.get_template())
# #     print(f"\n\n")
# response = query_engine.query("o cemu se radi")
# print(response)

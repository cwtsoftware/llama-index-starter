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

llm = OpenAI(temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context)

# # u ovom modulu će biti detaljnije objašnjen rad sa dokumentima, meta podacima i vektorskim indeksom
# # cilj je što bolje prilagoditi čvorove za što uspješnije i točnije upite što naposljetku
# # rezultira preciznijim odgovorima

# # tijekom učitavanja se dokumenti rastavljaju na manje dijelove
# # za učitavanje pojedinačnih dokumenata koristi se input_files, s obzirom da se radi o listi,
# # moguće je dodati i veći broj dokumenata na ovaj način
# documents = SimpleDirectoryReader(
#     input_files=["./godisnje-izvjesce-2022-CA.pdf"]
# ).load_data()

# # za dokumente koji su unutar datoteka koristi se input_dir
# # input_dir će učitati sve dokumente koje prepoznaje, a moguće je i postaviti
# # ograničenje za tipove dokumenata koje treba učitati
# # ako se postavi recursive=True, učitavaju se i dokumenti u pod datotekama
# documents = SimpleDirectoryReader(
#     input_dir="./whitepapers", 
#     required_exts=[".pdf"], 
#     recursive=True
# ).load_data()

# # postavljaju se default meda podaci poput naziva dokumenta, od kud je učitan, tip dokumenta i sl.
# print([x.metadata for x in documents])

# # definira se indeks
# # VectorStoreIndex za default pretvara dokumente u numeričke zapise u duljinama po 1024 tokena
# # moguće je i prilagoditi veličinu
# # prema default postavkama koristi se openai za pretvaranje u numeričke zapise 
# index = VectorStoreIndex.from_documents(
#     documents
# )

# # kao što je već spomenuto tijekom upita se upit pretvara u numerički zapis te se prema tome traže
# # najsličniji rezultati. Prema default postavkama uzimaju se 2 najsličnija rezultata i oni 
# # se postavljaju kao kontekst te se nadodje postavljen upit i to dvoje se šalje llm-u
# # ponekad je dobro i povećati broj najsličnijh rezultata koji se proslijeđuju kao kontekst
# # similarity_top_k atribut je za definiranje broja rezultata koji se proslijeđuju
# query_engine = index.as_query_engine(similarity_top_k=4) # uzima se 4 najsličnija izvora
# response = query_engine.query("koje sustave dokazivanje koristi bitcoin u usporedbi sa solana?")
# print(response)

# # izvori
# for node in response.source_nodes:
#   print("----------------------------------------------------")
#   print("node.score -> ", node.score) # relevantnost izvora, 0 - 1, 1 je najveća relevantnost
#   print("node.text -> ", node.text) # tekst izvora
#   print("node.metadata -> ", node.metadata) # meta podaci izvora

# # moguće je i filtrirati izvore prema meta podacima, a to je vrlo korisno kada se meta podaci prilagode
# # prema dokumentima koji se koriste
# # postavljaju se meta podaci na temelju imena dokumenta
# # prilikom postavljanja prilagođenih meta podataka, treba uzeti u obzir da je dostupan isključivo
# # file_path argument
# def get_meta(file_path):
#     if "bitcoin" in file_path:
#         return {"tech": "blockchain", "coin": "btc"}
#     elif "Ethereum" in file_path:
#         return {"tech": "blockchain", "coin": "eth"}
#     elif "solana" in file_path:
#         return {"tech": "blockchain", "coin": "sol"}
#     else:
#         return {"tech": "blockchain"}

# documents = SimpleDirectoryReader(
#     input_dir="./whitepapers", 
#     required_exts=[".pdf"], 
#     recursive=True,
#     file_metadata=get_meta
# ).load_data()

# index = VectorStoreIndex.from_documents(
#     documents
# )

# # definira se filter po kojem se želi pretraživati
# from llama_index.vector_stores import MetadataFilters, ExactMatchFilter
# filters = MetadataFilters(
#     filters=[ExactMatchFilter(key="coin", value="btc")]
# )

# query_engine = index.as_query_engine(streaming=True, filters=filters)
# streaming_response = query_engine.query("koji se sustav dokazivanja koristi")
# streaming_response.print_response_stream()

# # izvori
# for node in streaming_response.source_nodes:
#   print("\n----------------------------------------------------")
#   print("node.score -> ", node.score) # relevantnost izvora, 0 - 1, 1 je najveća relevantnost
#   print("node.text -> ", node.text) # tekst izvora
#   print("node.metadata -> ", node.metadata) # meta podaci izvora

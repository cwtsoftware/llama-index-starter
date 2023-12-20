import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import OpenAI

# slijed događaja do rezultata
# učitavanje -> indeksiranje -> upit (rezultat)

# tijekom faze učitavnja se učitavaju potrebni dokumenti, potrebni resursi i llm

# prije početka upita, moguće je prilagoditi llm koji se koristi, najčešće je to
# izdanje modela, temperatura (regulator nasumičnosti i kreativnosti, 0 - 1, gdje je 1 najveća vrijednost)
# i veličina odgovora ili upita

# postavi veličinu upita
context_window = 4096
# postavi maksimalnu veličinu odgovora
num_output = 256

# definiraj llm
llm = OpenAI(
    temperature=0.1,  # temperatura
    model="gpt-3.5-turbo",  # ili bilo koja druga verzija primjerice gpt-4
    max_tokens=num_output,  # maksimalna veličina odgovora
    system_prompt="Respond in Croatian language",  # početni upit koji llm uvijek ima tijekom svakog upita
)

# ServiceContext je paket često korištenih resursa koji se koriste tijekom
# stadija indeksiranja i upita
service_context = ServiceContext.from_defaults(
    llm=llm,
    context_window=context_window,
    num_output=num_output,
)

# za učitavanje lokalnih dokumeneta koristi se SimpleDirectoryReader koji
# učitava popularne formate poput .pdf, .docx, .csv, .md, .jpg, .jpeg i druge
documents = SimpleDirectoryReader(
    input_files=["./godisnje-izvjesce-2022-CA.pdf"]
).load_data()

# slijedeći korak je kreiranje indeksa koji se koristi za izvlačenje
# konteksta i/ili znanja iz vlastitih dokumenata i/ili izvora
# za ovaj primjer će se koristiti godišnje izvješće Croatia Airlines-a za 2022. god.
# indeks je struktura podataka koja omogućuje brzo dohvaćanje relevantnog konteksta za korisnički upit.
# tijekom ovog tutorial-a koristit će se VectorStoreIndex
# VectorStoreIndex pohranjuje svaki čvor (eng. node) kao numerički zapis unutar vektorske baze podataka
# tijekom upita se i sami upit pretvara u numerički zapis, zatim se dohvaćaju najsličniji čvorovi i proslijeđuju prema llm-u
index = VectorStoreIndex.from_documents(
  documents, 
  service_context=service_context
)

# nakon kreiranja indeksa, praksa je da se kreirani vektorski zapis spremi u dediciranu vektorsku bazu podataka, ali
# u ovom tutorialu, radi jednostavnosti, svaki puta kada se pokrene python skripta se kreira novi vektorski zapis i ne sprema se
# neki od često korištenih vektorskih baza podataka su Chroma, Pinecone, Faiss...

# za dobivanje rezultata kreira se query_engine preko kreiranog indeksa i postavi se upit
query_engine = index.as_query_engine()
response = query_engine.query("Poslovnice u inozemstvu")
print(response)

# streaming
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Poslovnice u inozemstvu")
streaming_response.print_response_stream()

# ako je potrebno prilagoditi stream, moguće je na slijedeći način:
# for text in streaming_response.response_gen:
#     # napravi nešto sa tekstom dok se generira npr. print(text)
#     pass

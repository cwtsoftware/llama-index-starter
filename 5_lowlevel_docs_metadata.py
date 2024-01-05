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

# u ovom modulu će se prikazati low-level api koncepti upravljanja sa dokumentima i kreiranja meta podataka te
# kreiranje automatskih meta podataka 
# postoje SummaryExtractor (sažimanje čvora), QuestionsAnsweredExtractor (postavlja set pitanja
# za kontekst čvora), TitleExtractor (postavlja naziv čvora), EntityExtractor (izvlači entitete)
from llama_index.schema import MetadataMode
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor
)

# alat za automatsko postavljanje meta podataka
# treba biti oprezan sa ovim alatima jer koriste llm pozive stoga i nose trošak sa korištenjem
# može se postaviti jedan ili više alata
extractors_1 = [
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

extractors_2 = [
    SummaryExtractor(summaries=["self"], llm=llm),
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

# u nastavku je prikazan primjer samostalnog definiranja instance dokumenta i dužine čvorova 
# podsjetnik: SimpleDirectoryReader koristi veličinu čvorova od 1024 tokena, a ovdje će se
# prikazati kreiranje čvorova od 512 tokena
from pypdf import PdfReader
from llama_index import Document

# kreira se instanca dokumenta za svaku stranicu teksta te se nadodaju proizvoljni meta podaci
reader = PdfReader("./whitepapers/bitcoin.pdf")
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    documents.append(
        Document(
            text=text,
            metadata={
                "filename": "bitcoin.pdf", 
                "page_label": i + 1,
                "coin": "btc"
            },
        )
    )

# TokenTextSplitter služi za kreiranje čvorova; pružaju se dokumenti koje treba raščlaniti u 
# pojedinačne čvorove
from llama_index.node_parser import TokenTextSplitter
node_parser = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=64
)

# kreiraju se čvorovi veličine 512 tokena; treba primjetiti da je kreirano 16 čvorova od 
# ukupno 9 prijašnje kreiranih dokumenata te da svaki čvor ima pripadajuće meta podatke od
# izvorne stranice
custom_nodes = node_parser.get_nodes_from_documents(documents)

for node in custom_nodes:
    print("--------------")
    print(node.text)
    print(node.metadata)

# procesuiranje čvorova sa automatske meta podatke
from llama_index.ingestion import IngestionPipeline
pipeline = IngestionPipeline(transformations=[node_parser, *extractors_1])
nodes_1 = pipeline.run(nodes=custom_nodes, in_place=False, show_progress=True)
print(nodes_1[3].get_content(metadata_mode="all"))

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_2])
nodes_2 = pipeline.run(nodes=custom_nodes, in_place=False, show_progress=True)
print(nodes_2[3].get_content(metadata_mode="all"))

index1 = VectorStoreIndex(
    nodes_1
)

index2 = VectorStoreIndex(
    nodes_2
)

query_engine1 = index1.as_query_engine(streaming=True, similarity_top_k=1)
query_engine2 = index2.as_query_engine(streaming=True, similarity_top_k=1)
response1 = query_engine1.query("koji mehanizam dokazivanja se koristi")
response2 = query_engine2.query("koji mehanizam dokazivanja se koristi")
print("response1 -> ", response1)
print("response2 -> ", response2)

# izvori
for node in response1.source_nodes:
    print("\n----------------------------------------------------")
    print("node.score -> ", node.score) # relevantnost izvora, 0 - 1, 1 je najveća relevantnost
    print("node.text -> ", node.text) # tekst izvora
    print("node.metadata -> ", node.metadata) # meta podaci izvora

# izvori
for node in response2.source_nodes:
    print("\n----------------------------------------------------")
    print("node.score -> ", node.score) # relevantnost izvora, 0 - 1, 1 je najveća relevantnost
    print("node.text -> ", node.text) # tekst izvora
    print("node.metadata -> ", node.metadata) # meta podaci izvora

# zaključak: ako je potrebno samostalno definirati meta podatke, najčešće će za precizno dohvačanje relevantnih čvorova biti dovoljan QuestionsAnsweredExtractor
# dok će drugi alati za automatsko postavljanje meta podataka biti korišteni u specifičnim slučajevima
# jedan od tih bi mogao biti kada je žele postaviti dugački čvorovi, zbog nužnosti konteksta tijekom upita, SummaryExtractor bi mogao poslužiti